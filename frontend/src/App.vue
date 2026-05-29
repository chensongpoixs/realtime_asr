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
  if (msg.includes('处理')) processing.value = true
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
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
  background: #11111b;
  color: #cdd6f4;
  min-height: 100vh;
}

/* ─── App 布局 ─── */
.app {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}

header {
  text-align: center;
  margin-bottom: 24px;
}

header h1 {
  font-size: 28px;
  color: #89b4fa;
  letter-spacing: 2px;
}

header .subtitle {
  font-size: 13px;
  color: #585b70;
}

main {
  display: flex;
  gap: 20px;
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

/* ─── 响应式 ─── */
@media (max-width: 768px) {
  main {
    flex-direction: column;
  }

  .sidebar {
    width: 100%;
  }
}
</style>
