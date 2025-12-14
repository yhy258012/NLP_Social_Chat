<template>
  <div class="page">
    <History />
    <Chat />
  </div>
</template>

<script setup lang="ts">
import Chat from './Chat.vue';
import History from './History.vue';
</script>

<style scoped>
.page {
  display: flex;
  /* 关键：使用Flex布局 */
  height: 100vh;
  /* 占满整个屏幕高度 */
  width: 100vw;
  overflow: hidden;
  /* 防止整个页面滚动 */
  background-color: var(--app-bg);

}

* {
  margin: 0;
  padding: 0;
}

body,
html {
  margin: 0;
  padding: 0;
}
</style>