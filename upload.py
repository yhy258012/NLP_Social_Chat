import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login, upload_folder
import gc  # 引入垃圾回收

# ================= 配置区域 =================
# 1. Hugging Face 登录 Token (确保是 Write 权限)
# 如果你已经本地 login() 过，这里可以留空，或者取消注释填入
# login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxx")

# 2. 你的仓库 ID
REPO_ID = "heizige/Qwen2.5-Social-3B-NB-Chat"

# 3. 本地路径配置
BASE_MODEL_PATH = "./models/Qwen/Qwen2.5-3B-Instruct"  # 基座模型
ADAPTER_PATH = "./models/qwen_social_finetune_final"  # LoRA 适配器
MERGED_DIR = "./models/qwen_social_3b_merged_full"  # [临时] 合并后存放的干净目录


# ===========================================
def merge_and_upload():
    print(f"🚀 1. [低内存模式] 正在加载基座模型: {BASE_MODEL_PATH} ...")

    # === 修改点 1: 强制使用 CPU 加载，并开启 low_cpu_mem_usage ===
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",  # 强制 CPU，防止 GPU 显存碎导致的问题
        low_cpu_mem_usage=True,  # 降低加载时的内存消耗
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)

    print(f"🚀 2. 正在加载 LoRA 适配器: {ADAPTER_PATH} ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, device_map="cpu")

    print("🚀 3. 正在执行合并 (Merge and Unload)...")
    model = model.merge_and_unload()

    # 手动清理一下内存
    gc.collect()

    print(f"🚀 4. [关键] 正在以小分块保存模型到: {MERGED_DIR} ...")

    # === 修改点 2: 设置 max_shard_size="1GB" ===
    # 默认是 5GB，改成 1GB 可以极大降低保存时的内存压力
    model.save_pretrained(MERGED_DIR, max_shard_size="1GB", safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DIR)

    if os.path.exists("README.md"):
        import shutil
        shutil.copy("README.md", os.path.join(MERGED_DIR, "README.md"))
        print("✅ README.md 已复制")

    print(f"🚀 5. 开始上传到 HuggingFace: {REPO_ID} ...")

    upload_folder(
        folder_path=MERGED_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload merged Qwen2.5-Social-3B model (1GB shards)"
    )

    print("\n🎉🎉🎉 上传成功！")


if __name__ == "__main__":
    # 确保你已经登录
    login(token="xxxxxxxxxxxxxxxxxxxxxxxxx")  # 替换为你的 Hugging Face Token
    merge_and_upload()