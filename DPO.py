import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

model_id = "Qwen/Qwen2.5-7B-Instruct"
lora_path = "./sft_output/final_lora"
dpo_data_path = "./dpo_pairs.jsonl"

# 1. 配置 4bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("加载 4bit 模型...")
# 2. 加载 base 并挂载 SFT LoRA 作为 Policy Model
base = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base, lora_path)

# 3. 读取偏好对数据
def load_dpo_data(file_path):
    raw_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                raw_data.append({
                    "prompt": obj["prompt"],
                    "chosen": obj["chosen"],
                    "rejected": obj["rejected"]
                })
    return Dataset.from_list(raw_data)

# 加载训练集和验证集的偏好对
train_dataset = load_dpo_data("dpo_pairs_train.jsonl")
eval_dataset = load_dpo_data("dpo_pairs_val.jsonl")

print(f"训练集: {len(train_dataset)} 条, 验证集: {len(eval_dataset)} 条")

# 4. 训练参数（实例化，而不是传类）
training_args = DPOConfig(
    output_dir="./dpo_output",
    num_train_epochs=3, 
    learning_rate=2e-6, 
    beta=0.1, 
    bf16=True,
    per_device_train_batch_size=1,    # 防止 OOM，设小一点
    gradient_accumulation_steps=8,
    logging_steps=1,
    max_length=1024,
    max_prompt_length=512,
    gradient_checkpointing=True, # 节省显存
    eval_strategy="epoch",               # 每轮结束验证一次
    save_strategy="epoch",               # 每轮结束保存一次
    load_best_model_at_end=True,         # 开启早停：训练结束后，自动加载验证集 Loss 最低的那个模型！
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

print("开始训练")

# 5. 初始化 DPOTrainer
trainer = DPOTrainer(
    model=model,                      # 使用上面挂了 LoRA 的 policy model
    ref_model=None,                   # trl 会自动使用 base 的初始状态作为 reference
    args=training_args,               # 传入实例
    train_dataset=train_dataset,   # 训练集
    eval_dataset=eval_dataset,     # 验证集
    processing_class=tokenizer,       
)

trainer.train()

trainer.save_model("./dpo_output/final_dpo_lora")
print("结束训练")