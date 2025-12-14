import pandas as pd
import json
import os
import random
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
DEEPSEEK_API_KEY = "sk-xxxxxx"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 两个文件夹路径
DIR_FINETUNED = "D:/program/ai_program/nlp_end_done/evaluate/results/"  # 微调模型结果
DIR_BASE = "D:/program/ai_program/nlp_end_done/evaluate/results6/"  # Ollama模型结果
OUTPUT_DIR = "D:/program/ai_program/nlp_end_done/evaluate/win_rate_results/"  # 结果保存路径

# 场景文件映射 (微调文件名 : Ollama文件名)
FILE_PAIRS = {
    "多轮评估表_长辈.xlsx": "deepseek-v3.1_671b-cloud_多轮评估表_长辈.xlsx",
    "多轮评估表_女友.xlsx": "deepseek-v3.1_671b-cloud_多轮评估表_女友.xlsx",
    "多轮评估表_导师.xlsx": "deepseek-v3.1_671b-cloud_多轮评估表_导师.xlsx",
    "多轮评估表_陌生人.xlsx": "deepseek-v3.1_671b-cloud_多轮评估表_陌生人.xlsx",
    "多轮评估表_夫妻.xlsx": "deepseek-v3.1_671b-cloud_多轮评估表_夫妻.xlsx"
}

# 人设 Prompt 映射
ROLE_PROMPTS_MAP = {
    "长辈": "你是一个情商极高的工科学生。你现在的对话对象是你的【长辈】。请保持尊敬、亲切的态度，并使用幽默、搞笑感来活跃气氛。",
    "女友": "你是一个风趣幽默的工科学生。你现在的对话对象是你的【女友】。对话充满中国式幽默却又不失暧昧，适当反转。其他时候要有甜美的感觉。",
    "导师": "你是一个理工科研究生，情商很高，说话有分寸。你现在的对话对象是你的【导师】。整体风格要：尊敬、专业、礼貌为主，同时可以适度幽默、机智。",
    "陌生人": "你是一个机智、得体、有分寸感的工科学生。你现在的对话对象是你的【陌生人】。保持轻松、礼貌的态度，并使用高情商幽默来化解尴尬或拉近距离。",
    "夫妻": "你是一个情商在线、风趣暖心的伴侣。你现在的对话对象是你的【配偶】。对话充满生活烟火气，兼具幽默调侃与温柔包容。"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


class BattleJudge:
    def __init__(self):
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    def compare(self, system_prompt, query, response_a, response_b):
        """
        对比两个回复，返回: 'A', 'B', 或 'Tie'
        """
        # 1. 随机交换位置以消除位置偏差 (Position Bias)
        is_swapped = random.choice([True, False])

        reply_1 = response_b if is_swapped else response_a
        reply_2 = response_a if is_swapped else response_b

        prompt = f"""
你是一位专业的对话质量评估专家。请根据【角色设定】对比两段AI生成的回复。

【角色设定】
{system_prompt}

【用户提问】
{query}

【回复 1】
{reply_1}

【回复 2】
{reply_2}

请判断哪个回复更好地符合了角色设定（如语气、幽默感、情商、得体程度）。
- 如果【回复 1】明显更好，请选择 1。
- 如果【回复 2】明显更好，请选择 2。
- 如果两者水平相当或难以区分，请选择 0 (平局)。

请务必只返回 JSON 格式：
{{
    "winner": 1 或 2 或 0,
    "reason": "简短的理由"
}}
"""
        try:
            res = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = res.choices[0].message.content
            result = json.loads(content.replace("```json", "").replace("```", "").strip())

            winner_idx = result["winner"]
            reason = result["reason"]

            # 2. 映射回原始模型 (反解交换逻辑)
            final_winner = "Tie"
            if winner_idx == 0:
                final_winner = "Tie"
            elif winner_idx == 1:
                final_winner = "B" if is_swapped else "A"
            elif winner_idx == 2:
                final_winner = "A" if is_swapped else "B"

            return final_winner, reason

        except Exception as e:
            print(f"⚠️ 评判出错: {e}")
            return "Error", str(e)


def main():
    judge = BattleJudge()
    total_stats = []

    print("⚔️  开始模型胜率评估 (Fine-tuned VS Ollama) ...")

    for ft_file, base_file in FILE_PAIRS.items():
        path_ft = os.path.join(DIR_FINETUNED, ft_file)
        path_base = os.path.join(DIR_BASE, base_file)

        if not os.path.exists(path_ft) or not os.path.exists(path_base):
            print(f"⚠️ 文件缺失，跳过组合: {ft_file}")
            continue

        print(f"\n📂 正在对比场景: {ft_file.split('_')[1].replace('.xlsx', '')}")

        # 读取数据
        df_ft = pd.read_excel(path_ft)
        df_base = pd.read_excel(path_base)

        # DEBUG: 打印一下列名，确保没读错
        print(f"   -> 微调文件列名: {list(df_ft.columns)}")
        print(f"   -> Ollama文件列名: {list(df_base.columns)}")

        min_len = min(len(df_ft), len(df_base))
        df_ft = df_ft.iloc[:min_len]
        df_base = df_base.iloc[:min_len]

        results = []
        ft_wins = 0
        base_wins = 0
        ties = 0

        # 开始逐行 PK
        for i in tqdm(range(min_len), desc="PK进度"):
            role_name = df_ft.iloc[i].get("场景", "未知")
            sys_prompt = ROLE_PROMPTS_MAP.get(role_name, role_name)
            query = df_ft.iloc[i].get("当前提问") or df_ft.iloc[i].get("用户提问")

            # === 核心修改：增加读取 Ollama 列名的逻辑 ===
            resp_ft = df_ft.iloc[i].get("【模型回复】") or \
                      df_ft.iloc[i].get("【模型生成的回复】")

            # 这里必须包含 "【Ollama模型回复】"
            resp_base = df_base.iloc[i].get("【Ollama模型回复】") or \
                        df_base.iloc[i].get("【原始模型回复】") or \
                        df_base.iloc[i].get("【模型回复】")

            # 如果任何一个为空，跳过
            if pd.isna(resp_ft) or pd.isna(resp_base):
                # 只有在第一行的时候报个错，提示一下
                if i == 0:
                    print(f"⚠️ 警告: 第一行数据读取为空! FT: {str(resp_ft)[:10]} | Base: {str(resp_base)[:10]}")
                continue

            # === 调用裁判 ===
            winner, reason = judge.compare(sys_prompt, query, resp_ft, resp_base)

            if winner == "A":
                ft_wins += 1
                win_label = "微调模型胜"
            elif winner == "B":
                base_wins += 1
                win_label = "Ollama模型胜"
            else:
                ties += 1
                win_label = "平局"

            results.append({
                "场景": role_name,
                "提问": query,
                "【微调模型回复】": resp_ft,
                "【Ollama模型回复】": resp_base,
                "PK结果": win_label,
                "裁判理由": reason
            })

        # 胜率计算
        total = ft_wins + base_wins + ties
        win_rate = (ft_wins / total) * 100 if total > 0 else 0

        print(f"📊 {role_name} 结果: 微调胜 {ft_wins} | Ollama胜 {base_wins} | 平局 {ties}")
        print(f"🏆 微调模型胜率: {win_rate:.2f}%")

        # 保存结果
        df_out = pd.DataFrame(results)
        out_path = os.path.join(OUTPUT_DIR, f"PK_Ollama_{role_name}.xlsx")
        df_out.to_excel(out_path, index=False)

        total_stats.append({
            "场景": role_name,
            "总场次": total,
            "微调胜": ft_wins,
            "Ollama胜": base_wins,
            "平局": ties,
            "微调胜率(%)": round(win_rate, 2)
        })

    print("\n" + "=" * 30)
    print("🌍 全局评估总结")
    print("=" * 30)
    df_stats = pd.DataFrame(total_stats)
    print(df_stats.to_string(index=False))

    stats_path = os.path.join(OUTPUT_DIR, "Ollama胜率总榜.xlsx")
    df_stats.to_excel(stats_path, index=False)
    print(f"\n✅ 所有评估完成，总榜已保存至: {stats_path}")


if __name__ == "__main__":
    main()
