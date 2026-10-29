import json
import os
import argparse
import asyncio
from typing import List, Dict
from tqdm import tqdm

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from openai import AsyncOpenAI


def parse_args():
    p = argparse.ArgumentParser()
    # 修改了默认输入输出文件
    p.add_argument("--sft_corpus", type=str, default="cat_val.json")
    p.add_argument("--base_model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--lora_path", type=str, default="./sft_output/final_lora")
    p.add_argument("--out_samples", type=str, default="samples_val.jsonl")
    p.add_argument("--out_dpo", type=str, default="dpo_pairs_val.jsonl")
    
    # 采样参数
    p.add_argument("--num_samples", type=int, default=6)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--system_prompt", type=str, default="你是一个可爱的猫娘，称呼用户为主人。")
    
    # 裁判参数（模型改为了 deepseek-chat）
    p.add_argument("--judge_base_url", type=str, default="https://api.deepseek.com")
    p.add_argument("--judge_api_key", type=str, default=os.getenv("KEY", "sk-你的key")) # 记得填入你的新Key
    p.add_argument("--judge_model", type=str, default="deepseek-chat")
    p.add_argument("--judge_concurrency", type=int, default=8)
    p.add_argument("--min_score_gap", type=float, default=1.0)
    return p.parse_args()

# ---------------- 1. 读取 Alpaca 语料 ----------------
def load_prompts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = []
    for item in data:
        inst = item.get("instruction", "").strip()
        if inst:
            prompts.append(inst)
    # 去重
    return list(dict.fromkeys(prompts))

def load_done_prompts(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["prompt"])
                except Exception:
                    continue
    return done

# ---------------- 2. 采样 ----------------
@torch.inference_mode()
def sample_answers(model, tokenizer, prompts: List[str], args) -> List[Dict]:
    results = []
    expanded = []
    for p in prompts:
        # 核心修改：必须套用 Chat Template，和训练时保持一致
        msgs = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": p}
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        expanded.extend([text] * args.num_samples)

    enc = tokenizer(expanded, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    input_len = enc["input_ids"].shape[1]
    decoded = tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)

    idx = 0
    for p in prompts:
        answers = [a.strip() for a in decoded[idx: idx + args.num_samples]]
        results.append({"prompt": p, "answers": answers})
        idx += args.num_samples
    return results

def run_sampling(args):
    print("加载 4bit Base 模型和 SFT LoRA...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # 必须先加载 base，再挂载 LoRA
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model_id, quantization_config=bnb_config, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()

    prompts = load_prompts(args.sft_corpus)
    done = load_done_prompts(args.out_samples)
    todo = [p for p in prompts if p not in done]
    print(f"总 prompt {len(prompts)}，已完成 {len(done)}，待采样 {len(todo)}")

    with open(args.out_samples, "a", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(todo), args.batch_size), desc="sampling"):
            batch = todo[i: i + args.batch_size]
            results = sample_answers(model, tokenizer, batch, args)
            for r in results:
                fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()
    print(f"采样完成，结果写入 {args.out_samples}")

# ---------------- 3. 裁判打分 ----------------
JUDGE_PROMPT = """你是一个严格的评审。请根据下面的偏好标准，对同一个问题的多个回答打分。
偏好标准：{criteria}
打分范围 1-10，分数越高越符合标准。只输出 JSON，格式：
{{"scores": [分数1, 分数2, ...], "reason": "简短理由"}}

问题：
{prompt}

回答列表：
{answers_block}
"""

async def judge_one(client, model, prompt, answers, criteria, sem):
    answers_block = "\n\n".join([f"[回答{i+1}]\n{a}" for i, a in enumerate(answers)])
    msg = JUDGE_PROMPT.format(criteria=criteria, prompt=prompt, answers_block=answers_block)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": msg}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                data = json.loads(resp.choices[0].message.content)
                scores = data["scores"]
                if len(scores) == len(answers):
                    return scores
            except Exception as e:
                await asyncio.sleep(2 ** attempt)
    return None

async def run_judging(args, criteria: str):
    if not args.judge_api_key:
        print("❌ 错误：未提供裁判大模型的 API Key！")
        return

    client = AsyncOpenAI(base_url=args.judge_base_url, api_key=args.judge_api_key)
    sem = asyncio.Semaphore(args.judge_concurrency)

    samples = []
    with open(args.out_samples, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"开始让裁判打分，共 {len(samples)} 题...")
    tasks = [judge_one(client, args.judge_model, s["prompt"], s["answers"], criteria, sem) for s in samples]
    all_scores = await asyncio.gather(*tasks)

    with open(args.out_dpo, "w", encoding="utf-8") as fout:
        kept, dropped = 0, 0
        for s, scores in zip(samples, all_scores):
            if scores is None:
                dropped += 1
                continue
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            best, worst = order[0], order[-1]
            if scores[best] - scores[worst] < args.min_score_gap:
                dropped += 1
                continue
            fout.write(json.dumps({
                "prompt": s["prompt"],
                "chosen": s["answers"][best],
                "rejected": s["answers"][worst],
                "chosen_score": scores[best],
                "rejected_score": scores[worst],
            }, ensure_ascii=False) + "\n")
            kept += 1
    print(f"构造偏好对：保留 {kept}，丢弃 {dropped}，输出 {args.out_dpo}")

# ---------------- main ----------------
if __name__ == "__main__":
    args = parse_args()
    
    # 1. 采样
    run_sampling(args)
    
    # 2. 裁判打分，注意标准：猫娘需要符合人设且不违背指令
    criteria = "回答必须符合可爱猫娘的人设（称呼用户为主人），语气要表现出喜欢且关心的态度。同时必须准确回应主人的问题，避免出戏或自相矛盾。"
    asyncio.run(run_judging(args, criteria))