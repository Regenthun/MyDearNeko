import os
import json
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 配置区 =================
# 请确保这两个路径是你刚才最新跑出来的日志路径
# (如果路径不对，请去文件夹里找一下 checkpoint 下的 trainer_state.json)
sft_log_path = "./sft_output/checkpoint-100/trainer_state.json"  # 替换成你实际的 SFT 路径
dpo_log_path = "./dpo_output/checkpoint-141/trainer_state.json"   # 替换成你实际的 DPO 路径

out_dir = "./evaluation/plots"
# =========================================

def read_logs(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return None
    with open(file_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    
    logs = state.get("log_history", [])
    data = {"step": [], "loss": [], "eval_loss": [], "eval_step": [], "grad_norm": [], "margin": []}
    
    step = 0
    for log in logs:
        if "loss" in log:
            step += 1
            data["step"].append(step)
            data["loss"].append(log["loss"])
            if "grad_norm" in log:
                data["grad_norm"].append(log["grad_norm"])
            if "rewards/margins" in log:
                data["margin"].append(log["rewards/margins"])
        elif "eval_loss" in log:
            # 记录验证集的步数（按比例估算，通常对应 epoch 结束）
            data["eval_step"].append(step)
            data["eval_loss"].append(log["eval_loss"])
            
    return data

def plot_all():
    os.makedirs(out_dir, exist_ok=True)
    sft_data = read_logs(sft_log_path)
    dpo_data = read_logs(dpo_log_path)

    # 图 1：SFT Loss 曲线
    if sft_data:
        plt.figure(figsize=(10, 4))
        plt.plot(sft_data["step"], sft_data["loss"], label="Train Loss", color="blue", alpha=0.6)
        if sft_data["eval_loss"]:
            plt.plot(sft_data["eval_step"], sft_data["eval_loss"], label="Val Loss", color="red", marker="o", linestyle="--")
        plt.title("SFT Training & Validation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/1_sft_loss.png", bbox_inches='tight', dpi=200)
        plt.close()
        print("✅ 已生成 SFT Loss 曲线")

    # 图 2：DPO Loss 曲线
    if dpo_data:
        plt.figure(figsize=(10, 4))
        plt.plot(dpo_data["step"], dpo_data["loss"], label="Train Loss", color="purple", alpha=0.6)
        if dpo_data["eval_loss"]:
            plt.plot(dpo_data["eval_step"], dpo_data["eval_loss"], label="Val Loss", color="orange", marker="o", linestyle="--")
        plt.title("DPO Training & Validation Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/2_dpo_loss.png", bbox_inches='tight', dpi=200)
        plt.close()
        print("✅ 已生成 DPO Loss 曲线")

    # 图 3：DPO 监控指标 (Grad Norm & Margin)
    if dpo_data:
        fig, ax1 = plt.subplots(figsize=(10, 4))
        
        # Grad Norm
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Grad Norm', color='tab:red')
        ax1.plot(dpo_data["step"], dpo_data["grad_norm"], color='tab:red', alpha=0.5, label="Grad Norm")
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        # Rewards Margin
        if dpo_data["margin"]:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Rewards Margin', color='tab:green')
            ax2.plot(dpo_data["step"], dpo_data["margin"], color='tab:green', label="Reward Margin")
            ax2.tick_params(axis='y', labelcolor='tab:green')
            
        plt.title("DPO Training Dynamics: Grad Norm & Reward Margin")
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{out_dir}/3_dpo_metrics.png", bbox_inches='tight', dpi=200)
        plt.close()
        print("✅ 已生成 DPO 监控指标曲线")

    print(f"\n🎉 所有图表已保存至 {out_dir} 文件夹！直接去那里拿图放进 README 即可。")

if __name__ == "__main__":
    plot_all()