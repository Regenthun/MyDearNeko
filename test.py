import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["OMP_NUM_THREADS"] = "4"

import json
import torch
import random
import asyncio
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from openai import AsyncOpenAI

model_id = "Qwen/Qwen2.5-7B-Instruct"
sft_lora_path = "./sft_output/final_lora"
dpo_lora_path = "./dpo_output/final_dpo_lora"
judge_api_key = "sk-你的key" # 🔴 记得替换你的 Key

# 读取你切分出来的独立测试集
test_file_path = "cat_test.json"
test_data_list = []  # 换个名字，因为要存字典了

with open(test_file_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)
    for item in test_data:
        prompt = item.get("instruction", "").strip()
        reference = item.get("output", "").strip() # 👈 提取标准回答
        if prompt:
            test_data_list.append({
                "prompt": prompt,
                "reference": reference
            })

print(f"成功加载独立测试集，共 {len(test_data_list)} 条数据。")

system_prompt = "你是一个可爱的猫娘，称呼用户为主人。"

def load_models():
    print("加载 Tokenizer 和 Base Model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    
    print("加载 Base 模型 (无微调)...")
    base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True)
    base_model.eval()
    
    print("挂载 SFT LoRA...")
    # 为了省显存，直接复用加载好的 base_model 挂载 SFT 测完后卸载，再挂 DPO。
    # 但 4bit 模型卸载比较麻烦，24GB 显存同时装三个模型可能会 OOM。
    # 稳妥起见，我们同时装 3 个。
    base_sft = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True)
    model_sft = PeftModel.from_pretrained(base_sft, sft_lora_path)
    model_sft.eval()
    
    print("挂载 DPO LoRA...")
    base_dpo = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True)
    model_dpo = PeftModel.from_pretrained(base_dpo, dpo_lora_path)
    model_dpo.eval()
    
    return tokenizer, base_model, model_sft, model_dpo

def generate_answer(model, tokenizer, prompt):
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=256, do_sample=True, 
            temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id
        )
    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

async def judge_three_way(client, prompt, reference, ans_base, ans_sft, ans_dpo):
    # 随机打乱三个回答的顺序，避免位置偏见
    answers = [ans_base, ans_sft, ans_dpo]
    models = ["Base", "SFT", "DPO"]
    combined = list(zip(answers, models))
    random.shuffle(combined)
    shuffled_answers, shuffled_labels = zip(*combined)
    
    judge_prompt = f"""你是一个严格的评委。请评估以下三个AI助手对用户问题的回答。
    
【参考标准回答】：{reference}

【评估标准】：
1. 准确性：回答的语义是否与参考标准回答一致，是否准确完成了用户指令（尤其是代码、翻译）。
2. 人设度：是否符合猫娘人设（称呼用户为主人，语气可爱且不失礼貌）。

用户问题：{prompt}

回答 1：{shuffled_answers[0]}
回答 2：{shuffled_answers[1]}
回答 3：{shuffled_answers[2]}

请分别给这三个回答打分（1-10分），分数越高越好。请只输出 JSON 格式：
{{"score_1": 分数, "score_2": 分数, "score_3": 分数, "reason": "对三个回答的简短评价"}}"""
    
    try:
        resp = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        res = json.loads(resp.choices[0].message.content)
        scores = [res["score_1"], res["score_2"], res["score_3"]]
        
        # 把分数映射回对应的模型
        model_scores = {"Base": 0, "SFT": 0, "DPO": 0}
        for i, label in enumerate(shuffled_labels):
            model_scores[label] = scores[i]
            
        return model_scores, res["reason"]
    except Exception as e:
        print(f"裁判打分失败: {e}")
        return {"Base": 0, "SFT": 0, "DPO": 0}, "Error"

async def main():
    tokenizer, base_model, model_sft, model_dpo = load_models()
    client = AsyncOpenAI(api_key=judge_api_key, base_url="https://api.deepseek.com")
    
    print("\n开始生成回答并进行带参考标准的盲评...")
    results = []
    total_scores = {"Base": 0, "SFT": 0, "DPO": 0}
    
    # 👇 遍历字典，把 prompt 和 reference 提取出来
    for item in tqdm(test_data_list):
        prompt = item["prompt"]
        reference = item["reference"]
        
        ans_base = generate_answer(base_model, tokenizer, prompt)
        ans_sft = generate_answer(model_sft, tokenizer, prompt)
        ans_dpo = generate_answer(model_dpo, tokenizer, prompt)
        
        # 👇 把 reference 也传给裁判
        scores, reason = await judge_three_way(client, prompt, reference, ans_base, ans_sft, ans_dpo)
        
        for m in total_scores:
            total_scores[m] += scores[m]
            
        results.append({
            "prompt": prompt,
            "reference": reference,  # 👈 存下来方便复盘
            "base_answer": ans_base,
            "sft_answer": ans_sft,
            "dpo_answer": ans_dpo,
            "scores": scores,
            "judge_reason": reason
        })
        
    # 👇 这里原本是 len(test_prompts)，要改成 len(test_data_list)
    total = len(test_data_list) 
    print("\n" + "="*40)
    print("🏆 三方评测结果汇总 (总分 / 平均分)")
    print(f"Base 模型: {total_scores['Base']} 分 / 平均 {total_scores['Base']/total:.2f}")
    print(f"SFT  模型: {total_scores['SFT']} 分 / 平均 {total_scores['SFT']/total:.2f}")
    print(f"DPO  模型: {total_scores['DPO']} 分 / 平均 {total_scores['DPO']/total:.2f}")
    print("="*40)
    
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/eval_three_way.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("详细对比结果已保存至 evaluation/eval_three_way.json")

if __name__ == "__main__":
    asyncio.run(main())