import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import json

model_id = "Qwen/Qwen2.5-7B-Instruct"

print("正在加载模型...")

#Tokenizer = AutoTokenizer.from_pretrained(model_id)

#Model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype = torch.bfloat16, device_map = "auto")

#显存爆了，采用4bit量化试试
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    #device_map="auto",
    trust_remote_code=True,
)

#这个地方不能直接用，发现cat.json语料集的格式是alpaca，我们需要用代码处理成sharegpt，
def load_and_format(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    
    dataset_list = []
    for i in original_data:
        dataset_list.append({
            "messages": [
                {"role": "system", "content": "你是一个可爱的猫娘，称呼用户为主人。"},
                {"role": "user", "content": i.get("instruction", "")},
                {"role": "assistant", "content": i.get("output", "")}
            ]
        })
    
    ds = Dataset.from_list(dataset_list)
    
    # 把 messages 转换成 trl 需要的 text 字段
    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    
    return ds.map(format_chat)

# 2. 分别读取训练集和验证集（注意改成你实际的文件名）
print("正在加载并处理数据集...")
train_dataset = load_and_format("cat_train.json")
eval_dataset = load_and_format("cat_val.json")

print(f"✅ 数据准备完毕！训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条")


#这里的作用层可以多一点，每层代表的作用和效果见笔记
#peft_Config = LoraConfig(r = 8, lora_alpha = 16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout = 0.05, task_type = "CAUSAL_LM")

model = prepare_model_for_kbit_training(model)

peft_Config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

#training_args = SFTConfig(output_dir = "./str_output", per_device_eval_batch_size=1, num_train_epochs=2, learning_rate=2e-4, bf16 = True, logging_steps=1, report_to=None)

training_args = SFTConfig(
    output_dir="./sft_output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=1e-4,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=5,
    report_to=None,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_seq_length=1024,
)

trainer = SFTTrainer(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args = training_args,
    processing_class = tokenizer,
    peft_config=peft_Config,
    model = model
)
print("开始 SFT 训练...")
trainer.train()

trainer.save_model("./sft_output/final_lora")
print("训练完成")