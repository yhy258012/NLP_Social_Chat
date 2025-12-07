import json
import os
import random  # 引入随机库

# ================= 配置区域 =================

# 1. 定义不同角色的“人设基调”
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
    '导师对话数据': (
        "你是一个理工科研究生，情商很高，说话有分寸。你现在的对话对象是你的【导师】。"
        "整体风格要：尊敬、专业、礼貌为主，同时可以适度幽默、机智，缓解科研和催稿带来的压力。"
        "面对导师的关心和提问，要耐心、具体地回答，体现你有认真思考和实际行动。"
        "面对催论文、催进度、批评指正时，先诚恳认领问题，再用轻松但不油腻的方式化解，"
        "可以自嘲、可以用技术类比（比如项目迭代、系统优化），但不要撒娇卖萌，也不要搞暧昧。"
        "记得多称呼“老师”，学会复述导师的建议并给出自己的下一步计划，"
        "既不卑微，也不过度顶嘴；如果要“怼回去”，要用高情商方式，比如用事实、数据或幽默反转，"
        "既守住学生的姿态，又不失风度。"
    ),
    '陌生人对话数据': (
        "你是一个机智、得体、有分寸感的工科学生。你现在的对话对象是你的【陌生人】。"
        "保持轻松、礼貌的态度，并使用高情商幽默来化解尴尬或拉近距离，"
        "对于冒犯或尴尬的问题要机智回应、保护隐私；对于无心的小误会要用幽默展现善意。"
        "当感觉投缘时，可以适度分享，用共同话题建立连接。"
        "当感觉不安全或对方意图不当时，礼貌地结束对话并离开。"
    ),
    '夫妻对话数据': (
        "你是一个情商在线、风趣暖心的伴侣。你现在的对话对象是你的【配偶】。"
        "对话充满生活烟火气，兼具幽默调侃与温柔包容，偶尔互怼却不伤人。"
        "对于日常琐事多换位思考，对于矛盾巧妙化解，对于关心加倍回应，用轻松语气传递爱意。"
    )
}

# 2. 输入文件列表
pri_data_list = [
    {
        'name': '长辈对话数据',
        'file': './data/elder_chat_list.json',
    },
    {
        'name': '女友对话数据',
        'file': './data/girl_chat_list.json',
    },
    {
        'name': '导师对话数据',
        'file': './data/teacher_chat_list.json',
    },
    {
        'name': '陌生人对话数据',
        'file': './data/stranger_chat_list.json',
    },
    {
        'name': '夫妻对话数据',
        'file': './data/wife_chat_list.json',
    }
]

# 3. 输出文件配置
OUTPUT_DIR = './data/train_test/'
TRAIN_FILE = os.path.join(OUTPUT_DIR, 'train.json')
TEST_FILE = os.path.join(OUTPUT_DIR, 'test.json')  # 纠正为 test.json，比较标准


# ================= 核心处理逻辑 =================


def process_single_file(input_path, dataset_name):
    """
    读取原始JSON，转换为 OpenAI 格式
    """
    if not os.path.exists(input_path):
        print(f"[跳过] 找不到文件: {input_path}")
        return []

    base_system_prompt = ROLE_SYSTEM_PROMPTS.get(dataset_name, "你是一个乐于助人的助手。")
    dataset_formatted_list = []

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        print(f"正在处理 [{dataset_name}]... 发现 {len(raw_data)} 条对话")

        for item in raw_data:
            scene = item.get('scene', '日常聊天')
            full_system_content = f"{base_system_prompt} 当前话题：【{scene}】。"

            messages = [{"role": "system", "content": full_system_content}]

            for turn in item.get('chat', []):
                raw_role = turn.get('role')
                content = turn.get('content')

                # 简单映射
                openai_role = "assistant" if raw_role == "我" else "user"

                messages.append({
                    "role": openai_role,
                    "content": content
                })

            dataset_formatted_list.append({"messages": messages})

        return dataset_formatted_list

    except Exception as e:
        print(f"[错误] 处理 {dataset_name} 时发生异常: {str(e)}")
        return []


# ================= 主程序入口 =================

if __name__ == "__main__":
    print("=== 开始数据转换与切分 (OpenAI 格式) ===")

    # 1. 收集所有数据
    all_combined_data = []
    for task in pri_data_list:
        data_chunk = process_single_file(task['file'], task['name'])
        if data_chunk:
            all_combined_data.extend(data_chunk)

    total_count = len(all_combined_data)
    print(f"--- 数据收集完毕，共 {total_count} 条 ---")

    if total_count > 0:
        # 2. 打乱数据顺序 (Shuffle) - 非常重要，保证训练集和测试集分布一致
        random.seed(42)  # 设置种子，保证每次运行结果一致（可复现）
        random.shuffle(all_combined_data)
        print("--- 数据已打乱顺序 ---")

        # 3. 计算切分点 (90% / 10%)
        split_index = int(total_count * 0.9)

        train_data = all_combined_data[:split_index]
        test_data = all_combined_data[split_index:]

        # 4. 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 5. 分别写入文件
        print(f"正在写入训练集: {TRAIN_FILE} ...")
        with open(TRAIN_FILE, 'w', encoding='utf-8') as f_train:
            json.dump(train_data, f_train, ensure_ascii=False, indent=2)

        print(f"正在写入测试集: {TEST_FILE} ...")
        with open(TEST_FILE, 'w', encoding='utf-8') as f_test:
            json.dump(test_data, f_test, ensure_ascii=False, indent=2)

        print("=== 任务完成 ===")
        print(f"训练集: {len(train_data)} 条")
        print(f"测试集: {len(test_data)} 条")

    else:
        print("警告：没有处理任何数据，请检查输入路径。")
