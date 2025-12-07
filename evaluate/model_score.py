import pandas as pd
import json
import os
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
DEEPSEEK_API_KEY = "sk-b7a9f81ab82d44dc8ce89e97257e2c71"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 1. 修改输入目录为 Ollama 结果目录
INPUT_DIR = "D:/program/ai_program/nlp_end_done/evaluate/results3/"

# 2. 修改为 Ollama 的文件名列表
FILES_TO_EVAL = [
    "Ollama模型_多轮评估表_长辈.xlsx",
    "Ollama模型_多轮评估表_女友.xlsx",
    "Ollama模型_多轮评估表_导师.xlsx",
    "Ollama模型_多轮评估表_陌生人.xlsx",
    "Ollama模型_多轮评估表_夫妻.xlsx"
]

# 3. 完整的 Prompt 映射 (评分标准)
ROLE_PROMPTS_MAP = {
    "长辈": "你是一个情商极高的工科学生。你现在的对话对象是你的【长辈】。请保持尊敬、亲切的态度，并使用幽默、搞笑感来活跃气氛。",
    "女友": "你是一个风趣幽默的工科学生。你现在的对话对象是你的【女友】。对话充满中国式幽默却又不失暧昧，适当反转。其他时候要有甜美的感觉。",
    "导师": "你是一个理工科研究生，情商很高，说话有分寸。你现在的对话对象是你的【导师】。整体风格要：尊敬、专业、礼貌为主，同时可以适度幽默、机智。",
    "陌生人": "你是一个机智、得体、有分寸感的工科学生。你现在的对话对象是你的【陌生人】。保持轻松、礼貌的态度，并使用高情商幽默来化解尴尬或拉近距离。",
    "夫妻": "你是一个情商在线、风趣暖心的伴侣。你现在的对话对象是你的【配偶】。对话充满生活烟火气，兼具幽默调侃与温柔包容。"
}


# ================= 评分逻辑 =================
class JudgeModel:
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def evaluate(self, system_prompt, user_query, model_response, reference):
        # 判空保护
        if not model_response or pd.isna(model_response):
            return {"score": 0, "reason": "错误：读取到的回复为空"}

        prompt = f"""
你是一位严格的角色扮演评估专家。请评估以下 AI 回复是否符合设定的人设。

【角色设定】
{system_prompt}

【用户提问】
{user_query}

【待评估回复】
{model_response}

【参考回复】
{reference}

请基于以下标准打分 (1-5分)：
1分：严重偏离人设，或逻辑错误。
2分：人设模糊，语气生硬。
3分：基本符合人设。
4分：人设鲜明，语气自然。
5分：完美演绎，情商极高。

请务必只返回 JSON 格式：
{{
    "score": 评分数字(整数),
    "reason": "简短理由"
}}
"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )
            content = response.choices[0].message.content
            clean_json = content.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"⚠️ API 请求失败: {e}")
            return {"score": 0, "reason": "API Error"}


# ================= 主程序 =================
def main():
    judge = JudgeModel()
    print(f"🚀 开始评估 Ollama 模型数据 (Results3)...")

    for filename in FILES_TO_EVAL:
        file_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(file_path):
            print(f"⚠️ 跳过: 找不到文件 {file_path}")
            continue

        print(f"\n📄 正在批改: {filename}")

        # 读取 Excel
        df = pd.read_excel(file_path)

        # 初始化评分列
        if "LLM评分" not in df.columns:
            df["LLM评分"] = ""
            df["LLM评语"] = ""

        scores = []

        # 遍历每一行
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="评分进度"):
            # 断点续传：如果有有效分数则跳过
            current_score = row.get("LLM评分")
            if pd.notna(current_score) and current_score != "" and isinstance(current_score,
                                                                              (int, float)) and current_score > 0:
                scores.append(current_score)
                continue

            # === 1. 获取人设 ===
            role_name = row.get("场景", "未知的场景")
            sys_prompt = ROLE_PROMPTS_MAP.get(role_name, role_name)

            # === 2. 关键修改：读取正确的列名 ===
            # 这里必须对应你生成脚本里写的列名 "【Ollama模型回复】"
            response = row.get("【Ollama模型回复】")

            # 兼容性 fallback
            if pd.isna(response):
                response = row.get("【模型回复】")

            # === Debug: 打印第一条看看是否读到了 ===
            if index == 0:
                print(f"[DEBUG] 正在读取: {str(response)[:50]}...")

            query = row.get("当前提问", row.get("用户提问"))
            reference = row.get("【参考回复】", row.get("【原始参考回复】"))

            # === 3. 调用 DeepSeek ===
            result = judge.evaluate(sys_prompt, query, response, reference)

            # 写入结果
            df.at[index, "LLM评分"] = result["score"]
            df.at[index, "LLM评语"] = result["reason"]

            if result["score"] > 0:
                scores.append(result["score"])

        # 保存回 Excel
        df.to_excel(file_path, index=False)

        # 打印平均分
        if scores:
            avg = sum(scores) / len(scores)
            print(f"✅ {filename} 平均分: {avg:.2f}")
        else:
            print(f"⚠️ {filename} 没有有效分数")


if __name__ == "__main__":
    main()