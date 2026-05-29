<template>
  <div class="transcription-box">
    <div class="header">
      <h3>转写结果</h3>
      <div class="header-actions">
        <span class="status" :class="{ connected: wsConnected }">
          {{ wsConnected ? '已连接' : '未连接' }}
        </span>
        <button @click="copyAll" class="btn-copy" v-if="fullText">复制</button>
        <button @click="clearAll" class="btn-clear" v-if="fullText">清除</button>
      </div>
    </div>

    <div class="content" ref="contentRef">
      <div v-if="!fullText && !currentPartial" class="placeholder">
        等待转写结果...
      </div>
      <div class="text-output">
        <span class="confirmed">{{ fullText }}</span>
        <span v-if="currentPartial" class="partial">{{ currentPartial }}</span>
        <span class="cursor" v-if="wsConnected || processing">|</span>
      </div>
    </div>

    <div v-if="statusMsg" class="status-bar">{{ statusMsg }}</div>
  </div>
</template>

<script setup>
import { ref, nextTick, watch } from 'vue'

const props = defineProps({
  wsConnected: { type: Boolean, default: false },
  processing: { type: Boolean, default: false },
  statusMsg: { type: String, default: '' },
})

const fullText = ref('')
const currentPartial = ref('')
const contentRef = ref(null)

function appendText(text) {
  fullText.value += (fullText.value ? ' ' : '') + text
  currentPartial.value = ''
  scrollToBottom()
}

function setPartial(text) {
  currentPartial.value = text
  scrollToBottom()
}

function clearAll() {
  fullText.value = ''
  currentPartial.value = ''
}

async function copyAll() {
  try {
    await navigator.clipboard.writeText(fullText.value)
  } catch {
    // fallback for older browsers
    const ta = document.createElement('textarea')
    ta.value = fullText.value
    document.body.appendChild(ta)
    ta.select()
    document.execCommand('copy')
    document.body.removeChild(ta)
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (contentRef.value) {
      contentRef.value.scrollTop = contentRef.value.scrollHeight
    }
  })
}

defineExpose({ appendText, setPartial, clearAll })
</script>

<style scoped>
.transcription-box {
  background: #1e1e2e;
  border-radius: 12px;
  padding: 20px;
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 300px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.header h3 {
  margin: 0;
  color: #cdd6f4;
  font-size: 16px;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status {
  font-size: 12px;
  padding: 3px 8px;
  border-radius: 10px;
  background: #45475a;
  color: #f38ba8;
}

.status.connected {
  background: #314f30;
  color: #a6e3a1;
}

.btn-copy,
.btn-clear {
  padding: 4px 10px;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  background: #45475a;
  color: #cdd6f4;
}

.btn-copy:hover,
.btn-clear:hover {
  background: #585b70;
}

.content {
  flex: 1;
  background: #11111b;
  border-radius: 8px;
  padding: 16px;
  overflow-y: auto;
  max-height: 400px;
  font-size: 16px;
  line-height: 1.8;
  color: #cdd6f4;
  word-break: break-word;
}

.placeholder {
  color: #585b70;
  font-style: italic;
}

.text-output {
  white-space: pre-wrap;
}

.confirmed {
  color: #cdd6f4;
}

.partial {
  color: #a6adc8;
  opacity: 0.7;
}

.cursor {
  color: #89b4fa;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  50% { opacity: 0; }
}

.status-bar {
  margin-top: 10px;
  padding: 6px 12px;
  background: #313244;
  border-radius: 6px;
  font-size: 12px;
  color: #a6adc8;
}
</style>
