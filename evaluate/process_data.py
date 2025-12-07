# 从测试集中分解出各个任务的数据，并保存为单独的文件

import json
import os

# 1. 定义文件路径配置
input_file_path = '../data/train_test/test_cleaned.json'
output_dir = './data'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 初始化分类容器
categorized_data = {
    "elder": [],  # 长辈/亲戚
    "girl": [],  # 女友
    "teacher": [],  # 导师
    "strange": [],  # 陌生人
    "wife": []  # 夫妻/配偶
}

# 3. 读取数据
try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f)
    print(f"成功读取源文件，共 {len(text_data)} 条数据。")
except FileNotFoundError:
    print(f"错误：找不到文件 {input_file_path}")
    text_data = []
except Exception as e:
    print(f"读取文件时发生错误: {e}")
    text_data = []

# 4. 遍历并分类数据
for item in text_data:
    try:
        # 获取第一条消息（System Prompt）的内容
        system_content = item['messages'][0]['content']

        # 根据关键词进行匹配
        if "【长辈】" in system_content or "【亲戚】" in system_content:
            categorized_data["elder"].append(item)

        elif "【女友】" in system_content:
            categorized_data["girl"].append(item)

        elif "【导师】" in system_content or "【老师】" in system_content:
            categorized_data["teacher"].append(item)

        elif "【陌生人】" in system_content:
            categorized_data["strange"].append(item)

        elif "【夫妻】" in system_content or "【配偶】" in system_content or "【妻子】" in system_content:
            categorized_data["wife"].append(item)

        else:
            # 这里的代码用于调试，查看是否有未匹配到的类型
            # print(f"未分类的数据类型: {system_content[:20]}...")
            pass

    except (KeyError, IndexError):
        print("跳过格式错误的数据项")
        continue

# 5. 定义保存映射关系
save_mapping = [
    ("elder", "elder_text.json"),
    ("girl", "girl_text.json"),
    ("teacher", "teacher_text.json"),
    ("strange", "strange_text.json"),
    ("wife", "wife_text.json")
]

# 6. 保存文件
print("-" * 30)
for key, filename in save_mapping:
    data_list = categorized_data[key]
    output_path = os.path.join(output_dir, filename)
    print(f"已保存 [{key:^8}] -> {output_path} (共 {len(data_list)} 条)")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        print(f"已保存 [{key:^8}] -> {output_path} (共 {len(data_list)} 条)")
    except Exception as e:
        print(f"保存 {filename} 失败: {e}")

print("-" * 30)
print("所有任务完成。")