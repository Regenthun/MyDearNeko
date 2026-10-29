# chat.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--lora_path", type=str, default="./sft_output/final_lora")
    p.add_argument("--system", type=str, default="你是一个傲娇的猫娘，称呼用户为主人。")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--history_turns", type=int, default=5, help="保留最近几轮对话作为上下文")
    return p.parse_args()


def main():
    args = parse_args()

    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id, quantization_config=bnb_config, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.eval()
    print("加载完成，输入 quit 退出。\n")

    # 对话历史（不含 system）
    history = []

    while True:
        try:
            user = input("主人: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见喵～")
            break
        if user.lower() in ("quit", "exit", "q"):
            print("再见喵～")
            break
        if not user:
            continue

        # 构造 messages：system + 最近 N 轮历史 + 当前 user
        messages = [{"role": "system", "content": args.system}]
        for u, a in history[-args.history_turns:]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user})

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        reply = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()

        print(f"猫娘: {reply}\n")
        history.append((user, reply))


if __name__ == "__main__":
    main()