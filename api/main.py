from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import torch
from threading import Thread
from transformers import TextIteratorStreamer
from model_loader import model_service, DEVICE
from fastapi.responses import StreamingResponse
import json

app = FastAPI(title="Qwen Social Chat API")

# 1. 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 角色映射表
ROLE_MAP = {
    1: "长辈",
    2: "女友",
    3: "导师",
    4: "陌生人",
    5: "夫妻"
}

# 3. 角色 Prompt 定义
ROLE_PROMPTS = {
    "长辈": "你是一个情商极高的工科学生。你现在的对话对象是你的【长辈】。请保持尊敬、亲切的态度，并使用幽默、搞笑感来活跃气氛。回复要自然，不要太长。",
    "女友": "你是一个风趣幽默的工科学生。你现在的对话对象是你的【女友】。对话充满中国式幽默却又不失暧昧，适当反转。其他时候要有甜美的感觉。多用口语，少说教。",
    "导师": "你是一个理工科研究生，情商很高，说话有分寸。你现在的对话对象是你的【导师】。整体风格要：尊敬、专业、礼貌为主，同时可以适度幽默、机智。回复要精炼。",
    "陌生人": "你是一个机智、得体、有分寸感的工科学生。你现在的对话对象是你的【陌生人】。保持轻松、礼貌的态度，并使用高情商幽默来化解尴尬或拉近距离。",
    "夫妻": "你是一个情商在线、风趣暖心的伴侣。你现在的对话对象是你的【配偶】。对话充满生活烟火气，兼具幽默调侃与温柔包容。多些关心，少些大道理。"
}


# 4. 请求体数据结构
class ChatRequest(BaseModel):
    role: int
    messages: List[Dict[str, str]]
    # 新增参数，允许前端微调，没有则使用默认值
    temperature: float = 0.85
    top_p: float = 0.95


# 5. 启动加载
@app.on_event("startup")
async def startup_event():
    model_service.load_model()


# 6. 核心聊天接口
@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    tokenizer, model = model_service.get_model()

    # --- A. 校验角色 ---
    role_name = ROLE_MAP.get(request.role)
    if not role_name:
        raise HTTPException(status_code=400, detail="无效的角色 ID")

    system_prompt = ROLE_PROMPTS[role_name]

    # --- [优化点 1]：上下文截断 (Context Truncation) ---
    # 如果历史记录太长，模型会“迷失”或显存溢出。保留最近的 N 轮对话效果最好。
    # 假设保留最近 10 轮 (20条消息)
    MAX_HISTORY_TURNS = 20
    recent_messages = request.messages[-MAX_HISTORY_TURNS:] if request.messages else []

    # --- B. 构建完整的对话历史 ---
    full_messages = [{"role": "system", "content": system_prompt}]

    for msg in recent_messages:
        # 严格过滤，防止前端传入错误的 system 导致 prompt 污染
        if msg['role'] in ['user', 'assistant']:
            full_messages.append(msg)

    print(f"当前角色: {role_name}")
    print(full_messages) # 调试时打开

    # --- C. 预处理输入 ---
    input_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(DEVICE)

    # --- D. 定义流式生成器 ---
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # --- [优化点 2]：调整生成参数 (Generation Config) ---
    # 这里的参数直接决定模型是“死板”还是“活泼”
    generation_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=512,

        # 1. Temperature (温度): 调高到 0.8-0.9 会更活泼、更有创造力；调低到 0.5 会更死板准确。
        # 社交闲聊建议 0.85
        temperature=request.temperature,

        # 2. Top-P (核采样): 控制候选词范围，0.9-0.95 比较合适
        top_p=request.top_p,

        # 3. Top-K (新增): 限制只从概率最高的 K 个词里选，防止生成离谱的词。建议 50。
        top_k=50,

        # 4. Do Sample: 必须为 True 才能让上面的参数生效
        do_sample=True,

        # 5. Repetition Penalty: 重复惩罚。
        # 如果模型喜欢复读，设为 1.1 或 1.2。如果模型说话不通顺，设回 1.05 或 1.0。
        repetition_penalty=1.1,

        # 6. EOS Token: 确保模型知道什么时候该停嘴 (Qwen 的结束符 ID 通常可以在 tokenizer 中找到)
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # --- E. 返回 SSE 流 ---
    async def response_generator():
        generated_text = ""
        for new_text in streamer:
            # 清理特殊符号
            clean_text = new_text.replace("<|im_end|>", "").replace("<|im_start|>", "")

            if clean_text:
                generated_text += clean_text
                response_json = {
                    "role": "assistant",
                    "content": clean_text
                }
                yield f"data: {json.dumps(response_json, ensure_ascii=False)}\n\n"

        # 打印完整的生成结果用于后台调试
        # print(f"AI回复: {generated_text}")
        yield "data: [DONE]\n\n"

    return StreamingResponse(response_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)