import json
import os

# ================= 配置区域 =================
FILES_TO_CLEAN = [
    './data/train_test/train.json',
    './data/train_test/test.json'
]


def clean_single_file(file_path):
    """
    清洗单个 JSON 文件：去除 None 值，强制转字符串
    """
    if not os.path.exists(file_path):
        print(f"[跳过] 文件不存在: {file_path}")
        return

    print(f"正在处理: {file_path} ...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"[错误] 读取文件失败: {e}")
        return

    cleaned_data = []
    total_conversations = len(raw_data)
    dropped_msgs_count = 0
    empty_conv_count = 0

    for item in raw_data:
        messages = item.get('messages', [])
        valid_messages = []

        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content')

            # --- 清洗规则 1: 检查 content 是否为 None ---
            if content is None:
                dropped_msgs_count += 1
                continue  # 跳过这条坏消息

            # --- 清洗规则 2: 强制转换为字符串 ---
            if not isinstance(content, str):
                # 比如有时候 content 是数字 123
                content = str(content)

            # --- 清洗规则 3: 去除首尾空白字符 (可选) ---
            content = content.strip()

            # 如果清洗后内容为空字符串，也可以选择跳过（视需求而定，这里保留）
            # if content == "": continue

            valid_messages.append({
                "role": role,
                "content": content
            })

        # 如果这一轮对话里至少有一条有效消息，才保留
        if len(valid_messages) > 0:
            cleaned_data.append({
                "messages": valid_messages
            })
        else:
            empty_conv_count += 1

    # 生成新文件名，例如 train_cleaned.json
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    file_name_no_ext = os.path.splitext(base_name)[0]
    new_output_path = os.path.join(dir_name, f"{file_name_no_ext}_cleaned.json")

    # 保存清洗后的文件
    with open(new_output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"--- 清洗报告 ({base_name}) ---")
    print(f"  原始对话数: {total_conversations}")
    print(f"  清洗后对话数: {len(cleaned_data)}")
    print(f"  剔除坏消息数 (content=None): {dropped_msgs_count}")
    print(f"  剔除空对话组: {empty_conv_count}")
    print(f"  已保存至: {new_output_path}")
    print("-" * 30)


if __name__ == "__main__":
    print("=== 开始数据清洗流程 ===")
    for f in FILES_TO_CLEAN:
        clean_single_file(f)
    print("=== 所有任务完成 ===")