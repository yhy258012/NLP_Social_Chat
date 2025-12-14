<template>
  <div class="history-sidebar">
    <div class="new-chat-btn" @click="handleCreateNewChat">
      <span>+ 发起新对话</span>
    </div>

    <div class="history-list">
      <div v-for="item in history" :key="item.id" class="history-item" :class="{ active: item.id === currentChatId }"
        @click="selectChat(item.id)">
        <span class="title">{{ item.title }}</span>
        <span class="delete-icon" @click.stop="deleteChat(item.id, $event)">×</span>
      </div>

      <div v-if="history.length === 0" class="empty-tip">
        暂无历史记录
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';
import { useChat } from './useChat'; // 使用 composable 模块

const { history, currentChatId, createNewChat, selectChat, deleteChat, loadHistory } = useChat();

// --- 核心修复逻辑 ---
const handleCreateNewChat = () => {
  // 1. 找到当前正在选中的对话
  const currentChat = history.value.find(h => h.id === currentChatId.value);

  // 2. 判断：如果当前对话存在，且消息列表为空 (说明这已经是一个新对话了)
  if (currentChat && currentChat.messages.length === 0) {
    // 直接返回，不执行创建操作
    return;
  }

  // 3. 只有当前对话有内容时，才创建新的
  createNewChat();
};

onMounted(() => {
  loadHistory();
  if (history.value.length === 0) {
    createNewChat();
  } else if (!currentChatId.value) {
    // @ts-ignore
    currentChatId.value = history.value[0].id;
  }
});
</script>

<style scoped>
/* 样式保持不变 */
.history-sidebar {
  width: 260px;
  height: 100%;
  background: linear-gradient(180deg, var(--surface-2), var(--surface));
  display: flex;
  flex-direction: column;
  padding: 14px;
  color: var(--text);
  border-right: 1px solid rgba(15, 23, 42, 0.06);
  box-shadow: 2px 0 12px rgba(15, 23, 42, 0.03);
}

.new-chat-btn {
  background: linear-gradient(90deg, rgba(79, 70, 229, 0.12), rgba(79, 70, 229, 0.06));
  padding: 10px 15px;
  border-radius: 12px;
  cursor: pointer;
  margin-bottom: 18px;
  transition: transform 0.12s ease;
  display: flex;
  align-items: center;
  font-size: 14px;
  color: var(--accent);
  border: 1px solid rgba(79, 70, 229, 0.08);
}

.new-chat-btn:hover {
  transform: translateY(-2px);
}

.history-list {
  flex: 1;
  overflow-y: auto;
}

.history-item {
  padding: 10px 15px;
  margin-bottom: 5px;
  border-radius: 20px;
  cursor: pointer;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.history-item:hover {
  background: var(--surface-2);
}

.history-item.active {
  background: linear-gradient(90deg, rgba(79, 70, 229, 0.12), rgba(79, 70, 229, 0.06));
  color: var(--accent);
}

.delete-icon {
  opacity: 0;
  font-weight: bold;
  padding: 0 5px;
}

.history-item:hover .delete-icon {
  opacity: 1;
}

.empty-tip {
  text-align: center;
  color: var(--muted);
  font-size: 12px;
  margin-top: 20px;
}
</style>