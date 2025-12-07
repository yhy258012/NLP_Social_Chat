import os
import torch
import json
import math
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback  # 引入回调基类
)
from peft import LoraConfig, get_peft_model, TaskType

# 强制禁用 BnB
os.environ["PEFT_FORCE_NO_BITSANDBYTES"] = "1"

# ================= 配置区域 =================
CONFIG = {
    "model_path": "./models/Qwen/Qwen2.5-3B-Instruct",
    "train_file": "./data/train_test/train_cleaned.json",
    "test_file": "./data/train_test/test_cleaned.json",
    "output_dir": "./models/qwen_social_finetune_final",
    "max_length": 1024,
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "num_epochs":4,
    "save_steps": 100,
    "logging_steps": 10,  # 图表采样的频率
}


# ================= 自定义进度条回调 =================
class PerEpochProgressCallback(TrainerCallback):
    """
    自定义回调：实现每个 Epoch 单独显示百分比进度
    """

    def __init__(self, total_epochs, steps_per_epoch):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch

    def on_step_end(self, args, state, control, **kwargs):
        # state.global_step 是当前总步数 (例如 50)
        # 计算当前在第几轮 (从1开始)
        current_step = state.global_step
        current_epoch = math.ceil(current_step / self.steps_per_epoch)

        # 计算当前轮内的步数 (1 ~ 98)
        steps_in_this_epoch = current_step % self.steps_per_epoch
        if steps_in_this_epoch == 0:
            steps_in_this_epoch = self.steps_per_epoch  # 整除说明刚好跑完这一轮

        # 计算百分比
        percentage = (steps_in_this_epoch / self.steps_per_epoch) * 100

        # 打印进度条 (覆盖同一行，实现动画效果)
        # 格式：[Epoch 1/3] 进度: 50/98 (51.02%) | Loss: xxxx

        # 获取最新的 loss (如果有)
        current_loss = "N/A"
        if state.log_history and "loss" in state.log_history[-1]:
            current_loss = f"{state.log_history[-1]['loss']:.4f}"

        print(
            f"\r🚀 [Epoch {current_epoch}/{self.total_epochs}] 进度: {steps_in_this_epoch}/{self.steps_per_epoch} ({percentage:.2f}%) | 最新Loss: {current_loss}",
            end="")


# ================= 绘图函数 =================
def plot_loss_curve(log_history, output_dir):
    steps = []
    losses = []
    for entry in log_history:
        if "loss" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

    if not steps:
        print("\n⚠️ 没有数据，无法绘图")
        return

    # 保存原始数据
    with open(os.path.join(output_dir, "training_logs.json"), "w", encoding="utf-8") as f:
        json.dump(log_history, f, indent=2)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='.', linestyle='-', color='#1f77b4', label='Training Loss')
    plt.title(f'Training Loss Curve (Epochs={CONFIG["num_epochs"]})')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    print(f"\n📈 Loss 曲线已保存至: {os.path.join(output_dir, 'loss_curve.png')}")


# ================= 主训练逻辑 =================
def train():
    print("🚀 正在初始化...")

    # 1. 准备 Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_path'], trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_path'],
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 2. 准备数据
    def process_func(example):
        input_ids = tokenizer.apply_chat_template(example['messages'], tokenize=True, truncation=True,
                                                  max_length=CONFIG['max_length'], add_generation_prompt=False)
        return {"input_ids": input_ids, "labels": input_ids, "attention_mask": [1] * len(input_ids)}

    dataset = load_dataset("json", data_files={"train": CONFIG['train_file'], "test": CONFIG['test_file']})
    tokenized_dataset = dataset.map(process_func, remove_columns=dataset["train"].column_names)

    # 3. 计算每轮步数 (为了进度条显示正确)
    num_train_samples = len(tokenized_dataset["train"])
    steps_per_epoch = math.ceil(num_train_samples / (CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']))
    print(f"📊 数据量: {num_train_samples} | 每轮步数: {steps_per_epoch} | 总轮数: {CONFIG['num_epochs']}")

    # 4. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=CONFIG['output_dir'],
            per_device_train_batch_size=CONFIG['batch_size'],
            gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
            learning_rate=CONFIG['learning_rate'],
            num_train_epochs=CONFIG['num_epochs'],
            save_steps=CONFIG['save_steps'],
            logging_steps=CONFIG['logging_steps'],
            fp16=True,
            optim="adamw_torch",
            ddp_find_unused_parameters=False,
            report_to="none",

            # 【关键】禁用默认的丑陋进度条，使用我们自己的
            disable_tqdm=True
        ),
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),

        # 【关键】注入我们写的进度条回调
        callbacks=[PerEpochProgressCallback(CONFIG['num_epochs'], steps_per_epoch)]
    )

    model.config.use_cache = False

    print("\n" + "=" * 40)
    print("🤖 开始训练 (按轮次显示进度)")
    print("=" * 40)

    trainer.train()

    print("\n\n✅ 训练完成！正在保存...")
    trainer.save_model(CONFIG['output_dir'])
    tokenizer.save_pretrained(CONFIG['output_dir'])

    # 绘制曲线
    plot_loss_curve(trainer.state.log_history, CONFIG['output_dir'])


if __name__ == "__main__":
    train()