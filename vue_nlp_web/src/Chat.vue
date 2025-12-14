<template>
    <div class="gemini-wrapper" :class="hasMessages ? 'layout-chat' : 'layout-center'">

       

        <div class="top-bar">
            <div class="status-dot">Social Chatbot</div>

        </div>

        <div class="content-area">

            <div v-if="!hasMessages" class="welcome-screen">
                <h1 class="gradient-title">嘿!<br>今天谁又惹你了？</h1>
            </div>

            <div v-else class="message-list" ref="msgRef">
                <div v-for="(msg, index) in currentMessages" :key="index" class="msg-row" :class="msg.role">

                    <div v-if="msg.role !== 'user'" class="avatar ai-avatar">
                        <svg viewBox="0 0 24 24" fill="currentColor">
                            <path
                                d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-4.5-9c.83 0 1.5-.67 1.5-1.5S8.33 8 7.5 8 6 8.67 6 9.5 6.67 11 7.5 11zm9 0c.83 0 1.5-.67 1.5-1.5S17.33 8 16.5 8 15 8.67 15 9.5 15.67 11 16.5 11zm-4.47 4.28l2.39 2.39c.39.39.39 1.02 0 1.41-.39.39-1.02.39-1.41 0l-2.39-2.39c-1.33-1.33-.39-3.61 1.41-1.41zM12 14c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
                        </svg>
                    </div>

                    <div class="bubble">
                        <div class="text-content" v-html="formatText(msg.content)"></div>
                    </div>
                </div>

                <div class="spacer"></div>
            </div>
        </div>

        <div class="input-dock">
            <div class="input-wrapper">

                <div class="role-selector">
                    <button v-for="role in roles" :key="role.id" class="role-chip"
                        :class="{ active: selectedRole === role.id }" @click="selectRole(role.id)">
                        <span class="icon">{{ role.icon }}</span>
                        <span class="label">{{ role.name }}</span>
                    </button>
                </div>

                <div class="input-box" :class="{ 'sending': isStreaming }">
                    <input v-model="inputContent" @keyup.enter="handleSend" placeholder="他/她跟你说什么了..."
                        :disabled="isStreaming" />
                    <button class="send-btn" @click="handleSend" :disabled="!inputContent.trim() || isStreaming">
                        <span v-if="!isStreaming">➤</span>
                        <span v-else class="loading-spin">⟳</span>
                    </button>
                </div>

                <p class="footer-note">V1.0.0</p>
            </div>
        </div>

    </div>
</template>

<script setup lang="ts">
import { ref, computed, nextTick, watch } from 'vue';
import { useChat } from './useChat'; // 使用 composable 模块

// --- 状态定义 ---
const { history, currentChatId, addMessage, createNewChat } = useChat(); // 沿用你的 Hook
const inputContent = ref('');
const msgRef = ref<HTMLElement | null>(null);
const isStreaming = ref(false);
const selectedRole = ref(1); // 默认为 1: 长辈

// 角色配置
const roles = [
    { id: 1, name: '长辈', icon: '👴' },

    { id: 3, name: '导师', icon: '🎓' },
    { id: 2, name: '女友', icon: '💕' },
    { id: 4, name: '陌生人', icon: '🕶️' },
    { id: 5, name: '夫妻', icon: '🏠' },
];

const getRoleName = (id: number) => roles.find(r => r.id === id)?.name || 'AI';

// --- 计算属性 ---
const currentMessages = computed(() => {
    if (!currentChatId.value) return [];
    const session = history.value.find(h => h.id === currentChatId.value);
    return session ? session.messages : [];
});

const hasMessages = computed(() => currentMessages.value.length > 0);

// --- 核心逻辑 ---

// 简单的文本格式化 (处理换行)
const formatText = (text: string) => text.replace(/\n/g, '<br>');

// 滚动到底部
const scrollToBottom = async () => {
    await nextTick();
    if (msgRef.value) {
        msgRef.value.scrollTop = msgRef.value.scrollHeight;
    }
};

const selectRole = (roleId: number) => {
    selectedRole.value = roleId;
    if (currentMessages.value.length)
        createNewChat()

};

// 发送并处理流式响应
const handleSend = async () => {
    const text = inputContent.value.trim();
    if (!text || isStreaming.value) return;

    // 0. 如果没有 ID (比如刚进页面)，需要先创建一个对话 ID (这里假设你的 useChat 会处理，或者手动赋值)
    if (!currentChatId.value) {
        // 简单模拟，具体看你的 useChat 实现
        // addSession(); 
        // currentChatId.value = Date.now(); 
    }

    // 1. UI 立即响应：用户消息上屏
    inputContent.value = '';
    addMessage(text, 'user');
    isStreaming.value = true;
    await scrollToBottom();

    // 2. 准备 AI 消息占位 (role 设为 'assistant' 或 'ai')
    // @ts-ignore
    addMessage('', 'assistant');
    // 获取刚刚添加的那条空消息的引用，以便后续追加内容
    // 注意：useChat 返回的可能是 reactive 对象，直接修改即可
    const session = history.value.find(h => h.id === currentChatId.value);
    // @ts-ignore
    const aiMsgIndex = session.messages.length - 1;

    try {
        // 3. 构建发送给后端的历史消息 (剔除 system，因为后端会加)
        // @ts-ignore
        const historyPayload = session.messages
            .slice(0, -1) // 去掉最后一条空的占位符
            .map(m => ({
                role: m.role === 'user' ? 'user' : 'assistant', // 确保 role 名称匹配后端
                content: m.content
            }));

        // 4. 发起 Fetch 请求
        const response = await fetch("http://localhost:8000/chat/completions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                role: selectedRole.value,
                messages: historyPayload
            })
        });

        if (!response.body) throw new Error("No response body");

        // 5. 读取流
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n\n");

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const jsonStr = line.replace("data: ", "").trim();
                    if (jsonStr === "[DONE]") break;
                    try {
                        const data = JSON.parse(jsonStr);
                        if (data.content) {
                            // --- 核心：追加内容到 Vue 的响应式数据中 ---
                            // @ts-ignore
                            session.messages[aiMsgIndex].content += data.content;
                            scrollToBottom(); // 实时滚动
                        }
                    } catch (e) { console.error(e); }
                }
            }
        }
    } catch (error) {
        console.error(error);
        // @ts-ignore
        session.messages[aiMsgIndex].content += "\n[连接断开，请检查后端]";
    } finally {
        isStreaming.value = false;
    }
};

watch(currentChatId, () => {
    nextTick(scrollToBottom);
});
</script>

<style scoped>
/* --- 1. 全局容器与布局 (Light theme) --- */
.gemini-wrapper {
    position: relative;
    width: 100%;
    height: 100%;
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    color: var(--text);
    font-family: 'Roboto', 'Helvetica Neue', sans-serif;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: var(--shadow);
    border-radius: 8px;
}

/* 布局A: 居中 (无消息) */
.layout-center .content-area {
    justify-content: center;
    align-items: center;
    padding-bottom: 120px;
}

.layout-center .input-wrapper {
    width: 60%;
    max-width: 800px;
}

/* 布局B: 对话 (有消息) */
.layout-chat .content-area {
    justify-content: flex-start;
}

.layout-chat .input-wrapper {
    width: 90%;
    max-width: 900px;
}

/* --- 2. 装饰元素 --- */
.ambient-glow {
    position: absolute;
    top: -30%;
    left: -10%;
    width: 140%;
    height: 140%;
    background: radial-gradient(circle at 30% 30%, rgba(79, 70, 229, 0.08), transparent 30%),
        radial-gradient(circle at 80% 80%, rgba(99, 102, 241, 0.06), transparent 30%);
    pointer-events: none;
    z-index: 0;
    filter: blur(28px);
}

.top-bar {
    color: var(--text);
    
    display: flex;
    align-items: center;
    padding: 0 20px;
    font-size: 13px;
    z-index: 2;
}

.status-text {
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 8px;
}

.status-dot {
    
    height: 10px;
    border-radius: 50%;
    font-weight: bold;
    color: #F86F69;
    box-shadow: 0 6px 14px rgba(79, 70, 229, 0.12);
    display: inline-block;
}

/* --- 3. 内容区域 --- */
.content-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    z-index: 1;
    position: relative;
    padding: 18px 0;
}

/* 欢迎语 */
.welcome-screen {
    text-align: left;
    animation: fadeIn 0.8s ease-out;
}

.gradient-title {
    font-size: 48px;
    font-weight: 600;
    line-height: 1.1;
    background: linear-gradient(90deg, var(--accent), #9B72CB, #F86F69);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
    margin: 0;
}

/* 消息列表 */
.message-list {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    width: 100%;
    max-width: 900px;
    margin: 0 auto;
    scroll-behavior: smooth;
}

.msg-row {
    display: flex;
    gap: 16px;
    margin-bottom: 18px;
    opacity: 0;
    animation: slideIn 0.28s forwards;
}

.msg-row.user {
    flex-direction: row-reverse;
}

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--accent);
    flex-shrink: 0;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
}

.avatar svg {
    width: 20px;
    height: 20px;
}

.bubble {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: 14px;
    line-height: 1.6;
    font-size: 15px;
    position: relative;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04);
}

.msg-row.user .bubble {
    background-color: var(--bubble-user);
    color: var(--text);
    border-bottom-right-radius: 8px;
}

.msg-row.assistant .bubble {
    background-color: var(--bubble-assistant);
    color: var(--text);
    padding-left: 0;
}

.role-label {
    font-size: 11px;
    color: #888;
    margin-bottom: 4px;
    text-transform: uppercase;
}

/* --- 4. 底部输入坞 --- */
.input-dock {
    width: 100%;
    background: transparent;
    padding-bottom: 20px;
    display: flex;
    justify-content: center;
    z-index: 10;
}

.input-wrapper {
    display: flex;
    flex-direction: column;
    gap: 12px;
    transition: width 0.5s ease;
}

/* 角色胶囊 */
.role-selector {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding: 4px 0;
}

/* 隐藏滚动条 */
.role-selector::-webkit-scrollbar {
    display: none;
}

.role-chip {
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    border: 1px solid rgba(15, 23, 42, 0.06);
    color: var(--text);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 13px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: all 0.18s;
    white-space: nowrap;
}

.role-chip:hover {
    transform: translateY(-2px);
}

.role-chip.active {
    background: linear-gradient(90deg, rgba(79, 70, 229, 0.12), rgba(79, 70, 229, 0.06));
    border-color: rgba(79, 70, 229, 0.2);
    color: var(--accent);
    box-shadow: 0 8px 24px rgba(79, 70, 229, 0.08);
}

/* 输入框 */
.input-box {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.8), rgba(250, 250, 250, 0.9));
    backdrop-filter: blur(6px);
    border-radius: 32px;
    height: 60px;
    display: flex;
    align-items: center;
    padding: 0 8px 0 20px;
    border: 1px solid rgba(15, 23, 42, 0.06);
    transition: all 0.25s;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
}

.input-box:focus-within {
    border-color: rgba(15, 23, 42, 0.12);
    box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}

.input-box.sending {
    border-color: rgba(79, 70, 229, 0.2);
}

input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text);
    font-size: 16px;
    outline: none;
}

input::placeholder {
    color: var(--muted);
}

.send-btn {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    border: none;
    background: var(--accent);
    color: #fff;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    transition: transform 0.15s ease;
}

.send-btn:hover:not(:disabled) {
    transform: translateY(-2px);
}

.send-btn:disabled {
    color: var(--muted);
    cursor: not-allowed;
    opacity: 0.6;
}

.footer-note {
    text-align: center;
    font-size: 11px;
    color: var(--muted);
    margin: 0;
}

/* --- 动画 --- */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.loading-spin {
    display: inline-block;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    100% {
        transform: rotate(360deg);
    }
}

/* 滚动条美化 */
.message-list::-webkit-scrollbar {
    width: 6px;
}

.message-list::-webkit-scrollbar-thumb {
    background: rgba(15, 23, 42, 0.08);
    border-radius: 6px;
}
</style>