def remove_bom(file_path):
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"已移除 {file_path} 的BOM标记")


# 使用示例
remove_bom('./data/wife_chat_list.json')