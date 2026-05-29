<template>
  <div class="app">
    <header>
      <h1>WhisperWeb</h1>
      <span class="subtitle">实时语音转文字</span>
    </header>

    <main>
      <aside class="sidebar">
        <ConfigPanel @config-changed="onConfigChanged" />
      </aside>

      <section class="main-content">
        <FileUpload
          @transcription-result="onTranscription"
          @status-change="onStatusChange"
        />
        <TranscriptionBox
          ref="transBoxRef"
          :ws-connected="wsConnected"
          :processing="processing"
          :status-msg="statusMsg"
        />
      </section>
    </main>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import ConfigPanel from './components/ConfigPanel.vue'
import FileUpload from './components/FileUpload.vue'
import TranscriptionBox from './components/TranscriptionBox.vue'

const transBoxRef = ref(null)
const wsConnected = ref(false)
const processing = ref(false)
const statusMsg = ref('')

function onTranscription(data) {
  if (transBoxRef.value) {
    if (data.partial) {
      transBoxRef.value.setPartial(data.text)
    } else {
      transBoxRef.value.appendText(data.text)
    }
  }
}

function onStatusChange(msg) {
  statusMsg.value = msg
  if (msg.includes('连接')) wsConnected.value = true
  if (msg.includes('转写完成') || msg.includes('失败') || msg.includes('错误')) {
    processing.value = false
    wsConnected.value = false
  }
  if (msg.includes('处理') || msg.includes('录音')) processing.value = true
}

function onConfigChanged(config) {
  console.log('配置已更新:', config)
}
</script>

<style>
/* ─── 全局样式 ─── */
*,
*::before,
*::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
    'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', Helvetica, Arial,
    sans-serif;
  background: #f5f5f5;
  color: #333;
  min-height: 100vh;
  -webkit-font-smoothing: antialiased;
}

/* ─── App 布局 ─── */
.app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

header {
  text-align: center;
  margin-bottom: 20px;
}

header h1 {
  font-size: 26px;
  color: #4a6cf7;
  letter-spacing: 1px;
}

header .subtitle {
  font-size: 13px;
  color: #999;
}

main {
  display: flex;
  gap: 20px;
  align-items: flex-start;
}

.sidebar {
  width: 320px;
  flex-shrink: 0;
}

.main-content {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-width: 0;
}

/* ─── 手机端响应式 ─── */
@media (max-width: 768px) {
  .app {
    padding: 12px;
  }

  header h1 {
    font-size: 22px;
  }

  main {
    flex-direction: column;
    gap: 12px;
  }

  .sidebar {
    width: 100%;
  }
}
</style>
