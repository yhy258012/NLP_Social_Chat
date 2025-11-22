import json
import os

# ================= 配置区域 =================

# 1. 定义不同角色的“人设基调” (System Prompt Template)
ROLE_SYSTEM_PROMPTS = {
    '长辈对话数据': (
        "你是一个情商极高的工科学生。你现在的对话对象是你的【长辈】。"
        "请保持尊敬、亲切的态度，并使用幽默、搞笑感来活跃气氛，"
        "对于关心和询问要耐心回答，对于催促或压力要巧妙化解。也可以直接怼回去。"
    ),
    '女友对话数据': (
        "你是一个风趣幽默的工科学生。你现在的对话对象是你的【女友】。"
        "对话充满中国式幽默却又不失暧昧，适当反转。"
        "对于一些无理要求可以适当怼她，其他时候要有甜美的感觉。"
    ),
}

# 2. 输入文件列表 (不再需要 target 字段，因为最后统一保存)
pri_data_list = [
    {
        'name': '长辈对话数据',
        'file': './data/elder_chat_list.json',
    },
    {
        'name': '女友对话数据',
        'file': './data/girl_chat_list.json',
    },
]

# 3. 最终合并输出的文件路径
FINAL_OUTPUT_PATH = './data/train_test/done_input.json'


# ================= 核心处理逻辑 =================

def process_single_file(input_path, dataset_name):
    """
    读取原始JSON，转换为ShareGPT格式列表并返回（不直接写入文件）
    """
    # 1. 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"[跳过] 找不到文件: {input_path}")
        return []

    # 2. 获取该数据集对应的基础人设
    base_system_prompt = ROLE_SYSTEM_PROMPTS.get(dataset_name, "你是一个乐于助人的助手。")

    dataset_formatted_list = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        print(f"正在处理 [{dataset_name}]... 发现 {len(raw_data)} 条对话")

        for item in raw_data:
            scene = item.get('scene', '日常聊天')

            # --- 构建动态 System Prompt ---
            full_system_content = f"{base_system_prompt} 当前话题：【{scene}】。"

            # --- 构建对话列表 ---
            conversations = [
                {
                    "from": "system",
                    "value": full_system_content
                }
            ]

            # --- 遍历并转换每一轮对话 ---
            for turn in item.get('chat', []):
                raw_role = turn.get('role')
                content = turn.get('content')

                if raw_role == "他/她":
                    sharegpt_role = "user"
                elif raw_role == "我":
                    sharegpt_role = "assistant"
                else:
                    sharegpt_role = "user"

                conversations.append({
                    "from": sharegpt_role,
                    "value": content
                })

            dataset_formatted_list.append({
                "conversations": conversations
            })

        return dataset_formatted_list

    except Exception as e:
        print(f"[错误] 处理 {dataset_name} 时发生异常: {str(e)}")
        return []


# ================= 主程序入口 =================

if __name__ == "__main__":
    print("=== 开始数据合并与转换 ===")

    # 1. 定义一个总列表，用于存放所有数据
    all_combined_data = []

    # 2. 循环处理每个任务，将结果加入总列表
    for task in pri_data_list:
        data_chunk = process_single_file(task['file'], task['name'])
        if data_chunk:
            all_combined_data.extend(data_chunk)  # 将列表合并
            print(f"   -> 已合并 {len(data_chunk)} 条数据来自 {task['name']}")

    # 3. 确保输出目录存在
    os.makedirs(os.path.dirname(FINAL_OUTPUT_PATH), exist_ok=True)

    # 4. 一次性写入最终的大文件
    print(f"正在写入最终文件: {FINAL_OUTPUT_PATH} ...")
    with open(FINAL_OUTPUT_PATH, 'w', encoding='utf-8') as f_out:
        json.dump(all_combined_data, f_out, ensure_ascii=False, indent=2)

    print(f"=== 所有任务完成 ===")
    print(f"最终数据集共包含: {len(all_combined_data)} 条数据")