import json
import random
import os
import argparse

def split_dataset(input_file, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    # 1. 校验比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("比例之和必须等于 1.0")

    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件：{input_file}")
        return

    # 2. 读取数据
    print(f"正在读取 {input_file} ...")
    with open(input_file, "r", encoding="utf-8") as f:
        # 兼容 JSON 数组和 JSONL 格式
        if input_file.endswith(".json"):
            data = json.load(f)
        else:
            data = [json.loads(line) for line in f if line.strip()]
            
    total_size = len(data)
    print(f"原始数据集总大小：{total_size} 条")

    # 3. 打乱数据 (极其关键！防止原始数据有顺序偏差)
    random.seed(seed)
    random.shuffle(data)

    # 4. 计算切分索引
    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)

    # 5. 切分数据
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 6. 生成输出文件名
    base_name = os.path.splitext(input_file)[0]
    ext = os.path.splitext(input_file)[1]
    
    out_train = f"{base_name}_train{ext}"
    out_val = f"{base_name}_val{ext}"
    out_test = f"{base_name}_test{ext}"

    # 7. 保存文件
    def save_file(filepath, dataset):
        with open(filepath, "w", encoding="utf-8") as f:
            if filepath.endswith(".json"):
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            else:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_file(out_train, train_data)
    save_file(out_val, val_data)
    save_file(out_test, test_data)

    print("\n✅ 数据切分完成！")
    print(f"  📘 训练集 (Train): {len(train_data)} 条 -> {out_train}")
    print(f"  📗 验证集 (Val):   {len(val_data)} 条 -> {out_val}")
    print(f"  📕 测试集 (Test):  {len(test_data)} 条 -> {out_test}")
    print(f"\n(注：使用了随机种子 seed={seed}，保证每次切分结果一致)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键切分数据集 (Train/Val/Test)")
    parser.add_argument("--input", type=str, default="cat.json", help="输入的原始文件路径 (默认 cat.json)")
    parser.add_argument("--train", type=float, default=0.7, help="训练集比例 (默认 0.7)")
    parser.add_argument("--val", type=float, default=0.15, help="验证集比例 (默认 0.15)")
    parser.add_argument("--test", type=float, default=0.15, help="测试集比例 (默认 0.15)")
    
    args = parser.parse_args()
    
    split_dataset(
        input_file=args.input, 
        train_ratio=args.train, 
        val_ratio=args.val, 
        test_ratio=args.test
    )