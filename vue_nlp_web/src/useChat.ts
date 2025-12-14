import { ref, watch } from 'vue';

// 定义数据接口
export interface Message {
    role: 'user' | 'ai' | 'assistant';
    content: string;
}

export interface ChatSession {
    id: string;
    title: string;
    messages: Message[];
    timestamp: number;
}

// 全局状态（单例模式，保证所有组件用的是同一份数据）
const history = ref<ChatSession[]>([]);
const currentChatId = ref<string>('');

export function useChat() {

    // 生成唯一ID
    const generateId = () => Date.now().toString();

    // 1. 初始化/加载数据
    const loadHistory = () => {
        const localData = localStorage.getItem('chat-history');
        if (localData) {
            history.value = JSON.parse(localData);
        }
    };

    // 2. 保存数据到 LocalStorage
    const saveHistory = () => {
        localStorage.setItem('chat-history', JSON.stringify(history.value));
    };

    // 3. 发起新对话
    const createNewChat = () => {
        const newId = generateId();
        const newSession: ChatSession = {
            id: newId,
            title: '新对话', // 初始标题
            messages: [],
            timestamp: Date.now(),
        };

        // 将新对话插入到最前面
        history.value.unshift(newSession);
        currentChatId.value = newId;
        //@ts-ignore
        const cSession: ChatSession[] = [];
        let t: boolean = true;
        history.value.forEach((s) => {
            if (s.title == '新对话') {
                if (t)
                    //@ts-ignore
                    cSession.push(s);
                t = false;

            } else {
                //@ts-ignore
                cSession.push(s);
            }
        });
        //@ts-ignore
        history.value = cSession;
        saveHistory();
    };

    // 4. 发送消息（添加消息并自动保存）
    const addMessage = (content: string, role: 'user' | 'ai' | 'assistant') => {
        const session = history.value.find(h => h.id === currentChatId.value);
        if (session) {
            session.messages.push({ role, content });

            //如果是用户的第一条消息，自动更新标题
            if (session.messages.length === 1 && role === 'user') {
                session.title = content.substring(0, 10) + (content.length > 10 ? '...' : '');
            }

            // 更新时间戳并重新排序，把最新的对话顶上去
            session.timestamp = Date.now();
            history.value.sort((a, b) => b.timestamp - a.timestamp);

            saveHistory();
        }
    };

    // 5. 切换/选择对话
    const selectChat = (id: string) => {
        currentChatId.value = id;
    };

    // 6. 删除对话（可选功能）
    const deleteChat = (id: string, e?: Event) => {
        e?.stopPropagation(); // 防止触发点击选择
        const index = history.value.findIndex(h => h.id === id);
        if (index !== -1) {
            history.value.splice(index, 1);
            // 如果删除的是当前选中的，重置选中状态或选中第一个
            if (id === currentChatId.value) {
                currentChatId.value = history.value[0]?.id || '';
            }
            saveHistory();
        }
    };

    // 监听变化自动保存（双重保险）
    watch(history, saveHistory, { deep: true });

    return {
        history,
        currentChatId,
        loadHistory,
        createNewChat,
        addMessage,
        selectChat,
        deleteChat
    };
}
