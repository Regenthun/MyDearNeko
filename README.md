# 🐱 MyDearNeko: Qwen2.5-7B 猫娘人设 SFT + DPO 偏好对齐实战

本项目基于 `Qwen2.5-7B-Instruct` 模型，通过完整的两阶段微调（SFT + DPO）实现了极具个性化的猫娘对话微调。项目涵盖了从数据清洗切分、QLoRA 显存优化、自动化裁判生成偏好数据，到小样本防过拟合训练与三方盲评的完整工业级闭环。
本项目采用的开源训练语料cat.json来自https://github.com/mindsRiverPonder/NekoLLM-practice/，十分感谢！没有您的开源语料就没有这个项目的今天！

## 📊 核心实验数据以及实验成果

为了科学评估对齐效果，我们从基础语料中独立切分了未参与训练的 **Test Set（测试集）**，并引入 LLM-as-a-Judge 机制，结合标准答案（Reference-based）进行了 Base / SFT / DPO 三方盲评：

| 模型阶段 | 平均得分 (1-10) | 表现总结 |
| :--- | :--- | :--- |
| **Base (Qwen2.5-7B)** | 5.03 | 无特定人设，直白回答问题 |
| **SFT (QLoRA)** | 7.10 | 成功注入猫娘人设，但偶尔出现指令遵循偏差 |
| **DPO (偏好对齐)** | **7.18 🌟** | **人设更稳定，兼顾猫娘语气与代码/翻译能力** |

<img width="1692" height="785" alt="1_sft_loss" src="https://github.com/user-attachments/assets/d279c379-2dee-40a4-b6ac-1b2fc63bf476" />

可以看到SFT训练过程中loss稳步下降，比较base模型有较大提升。

## 📈 训练监控与工程复盘 (Troubleshooting)

在本次实战中，记录并解决了大模型微调中最常见的几个问题：

### 1. 自动化偏好数据构造 (LLM-as-a-Judge)
*   没有现成的偏好对语料库
*   利用 SFT 模型对 Train Set 进行高温度（`T=0.9`）多路采样，并异步调用 DeepSeek API 对多路结果进行打分。
*   筛选最高分为 `chosen`，最低分为 `rejected`，自动过滤低区分度样本，生成高质量 DPO 偏好对。

### 2. 显存优化
*   第一次进行训练时直接OOM，了解到7B太大，针对单卡 24GB 显存限制，项目采用 **QLoRA (4-bit NormalFloat)** 结合 **Gradient Checkpointing**，成功在消费级显卡上完成 7B 模型的 SFT 与 DPO 训练。

### 3. DPO模型和SFT模型并未拉开明显差距，DPO训练时震荡幅度较大
<img width="1980" height="777" alt="3_dpo_metrics" src="https://github.com/user-attachments/assets/3671ae90-46d5-4a0b-a3ce-93504263943a" /><img width="1692" height="785" alt="2_dpo_loss" src="https://github.com/user-attachments/assets/371d3d2a-27c5-42b1-b089-5190cc9dfcfe" />
*   **现象表现**：最终盲评得分中，DPO 模型（7.18）仅略高于 SFT 模型（7.10），未达到预期的断层领先。同时，在当前的训练曲线中，DPO 的 Train Loss 与 Grad Norm 存在较大幅度的震荡。
*   **工程归因分析**：
    1.  **极端的 Batch Size**：受限于 24GB 单卡显存，DPO 训练时的 `per_device_train_batch_size` 被迫设为 1。在 DPO 的 Pairwise 对比损失计算中，单样本会导致极端的方差，从而引发梯度震荡。
    2.  **偏好边界模糊**：由于 SFT 阶段模型已经很好地掌握了猫娘人设，导致 DPO 自动构造出的 `chosen` 和 `rejected` 往往只在细节语气上有微小差异。区分度过小的样本使得模型难以进一步学习到显著特征。
*   **未来改进方向 (Future Work)**：
    1.  **引入梯度累积与显存优化**：后续计划使用 DeepSpeed ZeRO 或梯度累积技术，在显存允许的极限下增大有效 Batch Size，以平滑梯度。
    2.  **构建“负样本降级”策略**：在构造 DPO 数据时，除了用 SFT 采样，可以引入 Base 模型的输出作为 `rejected`，强行拉大 `chosen` 与 `rejected` 的差异度，迫使 DPO 学到更明显的人设边界。

## 📂 项目目录结构

```text
SFT-7B/
├── Split.py               # 数据集切分脚本 (Train/Val/Test = 7:1.5:1.5)
├── SFT.py                 # 监督微调脚本 (QLoRA)
├── SamplingAndJudge.py    # 偏好数据自动生成 Pipeline (异步 API 调用)
├── DPO.py                 # DPO 训练脚本 (含验证集早停)
├── eval_three_way.py      # Base/SFT/DPO 三方盲评测试
├── Plot_curve.py          # 训练日志解析与可视化
├── evaluation/            # 评测结果 JSON 与 训练曲线图表
├── cat.json               # 原始语料
└── chat.py                # 单轮对话测试脚本
