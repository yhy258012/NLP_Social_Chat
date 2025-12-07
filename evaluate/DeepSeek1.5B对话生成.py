import os
import json
import pandas as pd
from tqdm import tqdm
import ollama

# ================= 配置区域 =================
# Ollama 模型名称
OLLAMA_MODEL_NAME = "gpt-oss:20b"

# 文件路径配置
EVAL_DATA_DIR = "D:/program/ai_program/nlp_end_done/evaluate/data/"
OUTPUT_DIR = "D:/program/ai_program/nlp_end_done/evaluate/results4/"  # 修改为 result3 目录

SCENARIO_FILES = {
    "长辈": "elder_text.json",
    "女友": "girl_text.json",
    "导师": "teacher_text.json",
    "陌生人": "strange_text.json",
    "夫妻": "wife_text.json"
}

ROLE_PROMPTS = {
    "长辈": "你是一个情商极高的工科学生。你现在的对话对象是你的【长辈】。请保持尊敬、亲切的态度，并使用幽默、搞笑感来活跃气氛。",
    "女友": "你是一个风趣幽默的工科学生。你现在的对话对象是你的【女友】。对话充满中国式幽默却又不失暧昧，适当反转。其他时候要有甜美的感觉。",
    "导师": "你是一个理工科研究生，情商很高，说话有分寸。你现在的对话对象是你的【导师】。整体风格要：尊敬、专业、礼貌为主，同时可以适度幽默、机智。",
    "陌生人": "你是一个机智、得体、有分寸感的工科学生。你现在的对话对象是你的【陌生人】。保持轻松、礼貌的态度，并使用高情商幽默来化解尴尬或拉近距离。",
    "夫妻": "你是一个情商在线、风趣暖心的伴侣。你现在的对话对象是你的【配偶】。对话充满生活烟火气，兼具幽默调侃与温柔包容。"
}

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


class DeepChat:
    def __init__(self, model_name):
        self.model_name = model_name

    def chat(self, messages):
        """调用 Ollama 生成回复"""
        try:
            # options 可以设置 temperature 等参数，这里保持默认或根据需要调整
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 4096 # 确保上下文足够长
                }
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"⚠️ Ollama 调用出错: {e}")
            return "Error: Generation Failed"


def format_history_for_excel(messages):
    """格式化历史消息用于Excel展示"""
    text = ""
    for msg in messages:
        role = "AI" if msg['role'] == 'assistant' else "用户"
        if msg['role'] == 'system': continue
        text += f"[{role}]: {msg['content']}\n"
    return text.strip()


def main():
    # 初始化 Ollama 聊天类
    bot = DeepChat(OLLAMA_MODEL_NAME)
    print(f"🚀 已连接 Ollama 模型: {OLLAMA_MODEL_NAME}")

    for role_name, filename in SCENARIO_FILES.items():
        file_path = os.path.join(EVAL_DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}")
            continue

        print(f"\n🤖 正在逐轮评估场景 (Ollama): 【{role_name}】...")
        current_system_prompt = ROLE_PROMPTS.get(role_name, "你是一个乐于助人的助手。")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        excel_data = []

        # 遍历每一个对话 Session
        for session_idx, item in enumerate(tqdm(data, desc=f"处理 {role_name}")):
            messages = item['messages']

            # === 核心逻辑：遍历对话中的每一轮 ===
            for i in range(len(messages)):
                msg = messages[i]

                # 如果当前是 User 发言，且下一条是 AI 发言，这就是一个测试点
                if msg['role'] == 'user' and (i + 1 < len(messages)) and messages[i + 1]['role'] == 'assistant':

                    # 1. 截取到当前 User 的历史作为输入
                    # 注意：为了避免修改原数据，这里使用 copy
                    raw_slice = messages[:i + 1]
                    input_msgs = [dict(m) for m in raw_slice]

                    # 2. 强制注入 System Prompt
                    if input_msgs[0]['role'] == 'system':
                        input_msgs[0]['content'] = current_system_prompt
                    else:
                        input_msgs.insert(0, {"role": "system", "content": current_system_prompt})

                    # 3. 提取真值 (Ground Truth)
                    reference_answer = messages[i + 1]['content']

                    # 4. Ollama 模型生成
                    model_reply = bot.chat(input_msgs)

                    # 5. 计算当前是第几轮 (粗略计算)
                    turn_index = (i + 1) // 2

                    excel_data.append({
                        "对话ID": session_idx + 1,
                        "轮次": f"第 {turn_index} 轮",
                        "场景": role_name,
                        "对话历史 (Context)": format_history_for_excel(input_msgs[:-1]),
                        "当前提问": msg['content'],
                        "【Ollama模型回复】": model_reply,  # 列名区分
                        "【参考回复】": reference_answer,
                        "评分 (1-5)": ""
                    })

        # 保存 Excel
        df = pd.DataFrame(excel_data)
        # 文件名保持一致性，方便后续脚本读取
        save_path = os.path.join(OUTPUT_DIR, f"gpt-oss:20b模型_多轮评估表_{role_name}.xlsx")
        df.to_excel(save_path, index=False)
        print(f"✅ 表格已生成: {save_path}")


if __name__ == "__main__":
    main()