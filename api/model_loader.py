import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# 配置路径 (请确保路径正确)
BASE_MODEL_PATH = "../models/Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "../models/qwen_social_finetune_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ModelService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
        return cls._instance

    def load_model(self):
        if self.model is not None:
            return

        print("🚀 正在加载 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("🚀 正在加载基座模型 (FP16)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print(f"🚀 正在注入 LoRA 适配器: {ADAPTER_PATH}...")
        self.model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        print("✅ 模型加载完成！")

    def get_model(self):
        return self.tokenizer, self.model

# 全局单例
model_service = ModelService()