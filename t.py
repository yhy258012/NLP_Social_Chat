# 将 new_chat 中的每个聊天对象添加到原来的 data 列表中
data.extend(new_chat)

# 写回 JSON 文件
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("多条聊天记录已添加成功！")