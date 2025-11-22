import json

# 读取 JSON 文件
file_path = './data/elder_chat_list.json'
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)  # data 是一个列表

# new_chat 是一个数组，包含多个聊天对象
new_chat = data

for index, item in enumerate(new_chat):
    item['id']=index+1

print(new_chat[0]['id'])

print(len(new_chat))
data=[]
# 将 new_chat 中的每个聊天对象添加到原来的 data 列表中
data.extend(new_chat)

# 写回 JSON 文件
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("多条聊天记录已添加成功！")
