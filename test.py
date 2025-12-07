import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time

# ==================== 配置区域 ====================
# 你的原始基座模型路径 (从 Hugging Face 下载的)
BASE_MODEL_PATH = "./models/Qwen/Qwen2.5-3B-Instruct"

# 你微调生成的 LoRA 适配器路径 (根据你的截图，就是这个文件夹)
ADAPTER_PATH = "models/qwen_social_finetune_final"

# 显存配置 (保持和训练时一致的 FP16)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==================== 角色人设定义 ====================
# 注意：这些 SYSTEM PROMPT 必须和训练数据中的风格一致！

ROLE_PROMPTS = {
    "长辈": (
        "你是一个情商极高的工科学生。你现在的对话对象是你的【长辈】。"
        "请保持尊敬、亲切的态度，并使用幽默、搞笑感来活跃气氛。"
    ),
    "女友": (
        "你是一个风趣幽默的工科学生。你现在的对话对象是你的【女友】。"
        "对话充满中国式幽默却又不失暧昧，适当反转。其他时候要有甜美的感觉。"
    ),
    "导师": (
        "你是一个理工科研究生，情商很高，说话有分寸。你现在的对话对象是你的【导师】。"
        "整体风格要：尊敬、专业、礼貌为主，同时可以适度幽默、机智。"
    ),
    '陌生人': (
        "你是一个机智、得体、有分寸感的工科学生。你现在的对话对象是你的【陌生人】。"
        "保持轻松、礼貌的态度，并使用高情商幽默来化解尴尬或拉近距离，"
        "对于冒犯或尴尬的问题要机智回应、保护隐私；对于无心的小误会要用幽默展现善意。"
        "当感觉投缘时，可以适度分享，用共同话题建立连接。"
        "当感觉不安全或对方意图不当时，礼貌地结束对话并离开。"
    ),
    '夫妻': (
        "你是一个情商在线、风趣暖心的伴侣。你现在的对话对象是你的【配偶】。"
        "对话充满生活烟火气，兼具幽默调侃与温柔包容，偶尔互怼却不伤人。"
        "对于日常琐事多换位思考，对于矛盾巧妙化解，对于关心加倍回应，用轻松语气传递爱意。"
    )
}


def load_model_and_adapter():
    """加载基座模型和LoRA适配器"""
    print("--- 1. 正在加载 Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        padding_side="left"  # 推理时通常使用 left padding
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--- 2. 正在加载 FP16 基座模型 ---")
    # 注意: 加载时必须使用 FP16，因为训练时的权重也是在 FP16 模型的内存上计算的
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",  # 自动分配到 GPU
        trust_remote_code=True
    )

    print(f"--- 3. 正在加载 LoRA 适配器: {ADAPTER_PATH} ---")
    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        torch_dtype=torch.float16,
    )

    # 启用评估模式 (禁用 Dropout 等)
    model.eval()

    # 打印模型结构，确认 LoRA 注入成功
    print("\n✅ 模型和适配器加载成功！")
    return tokenizer, model


def generate_response(tokenizer, model, scenario, user_input):
    """根据场景和输入生成回复"""

    # 1. 构造 OpenAI 格式的 Messages
    messages = [
        {"role": "system", "content": ROLE_PROMPTS[scenario]},
        {"role": "user", "content": user_input}
    ]

    # 2. 应用 Qwen 模板并 Tokenize
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # 告诉模型：接下来该你说话了
    )

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(DEVICE)

    # 3. 生成配置
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            do_sample=True,  # 开启采样，让回答更具创造性
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1  # 避免重复
        )

    # 4. 解码并清理输出
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # Qwen 的输出需要特殊处理，提取助手回复部分
    # 找到最后一个 Assistant 的标记，并清理后面的 EOS/IM_END 标记
    if "<|im_start|>assistant" in output_text:
        assistant_start_index = output_text.rfind("<|im_start|>assistant")
        assistant_reply = output_text[assistant_start_index:].replace("<|im_start|>assistant\n", "").strip()
        assistant_reply = assistant_reply.replace("<|im_end|>", "").strip()

        # 移除可能重复的 system prompt
        for prompt_text in ROLE_PROMPTS.values():
            if assistant_reply.startswith(prompt_text):
                assistant_reply = assistant_reply.replace(prompt_text, "").strip()

        return assistant_reply
    else:
        return output_text  # 返回完整输出以便调试


if __name__ == "__main__":
    if not os.path.exists(ADAPTER_PATH):
        print(f"❌ 错误: 找不到适配器路径 {ADAPTER_PATH}")
        print("请确认 ADAPTER_PATH 和 BASE_MODEL_PATH 配置是否正确。")
    else:
        # 计时器
        start_time = time.time()

        tokenizer, model = load_model_and_adapter()

        load_time = time.time() - start_time
        print(f"\n模型总加载耗时: {load_time:.2f} 秒")

        print("\n" + "=" * 50)
        print("🤖 开始多角色风格测试 🤖")
        print("=" * 50)

        # --- 测试案例 1: 长辈 ---
        scenario = "长辈"
        prompt = "年纪轻轻的不懂得尊老爱幼吗？"
        print(f"--- 场景: {scenario} ---")
        print(f"提问: {prompt}")
        response = generate_response(tokenizer, model, scenario, prompt)
        print(f"回复: {response}\n")

        # --- 测试案例 2: 女友 ---
        scenario = "女友"
        prompt = "你是不是在想前女友呢？"
        print(f"--- 场景: {scenario} ---")
        print(f"提问: {prompt}")
        response = generate_response(tokenizer, model, scenario, prompt)
        print(f"回复: {response}\n")

        # --- 测试案例 3: 导师 ---
        scenario = "导师"
        prompt = "你干什么吃的，这都不会？"
        print(f"--- 场景: {scenario} ---")
        print(f"提问: {prompt}")
        response = generate_response(tokenizer, model, scenario, prompt)
        print(f"回复: {response}\n")

        # --- 测试案例 4: 陌生人 ---
        scenario = "陌生人"
        prompt = "唉，你看那个人穿得好奇怪啊。"
        print(f"--- 场景: {scenario} ---")
        print(f"提问: {prompt}")
        response = generate_response(tokenizer, model, scenario, prompt)
        print(f"回复: {response}\n")

        # --- 测试案例 5: 夫妻 ---
        scenario = "夫妻"
        prompt = "我看你这就是懒，这点家务都不想做！"
        print(f"--- 场景: {scenario} ---")
        print(f"提问: {prompt}")
        response = generate_response(tokenizer, model, scenario, prompt)
        print(f"回复: {response}\n")