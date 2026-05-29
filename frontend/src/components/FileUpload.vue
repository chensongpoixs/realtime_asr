<template>
  <div class="file-upload">
    <!-- 文件上传模式 -->
    <div class="upload-zone"
         @drop.prevent="handleDrop"
         @dragover.prevent="dragover = true"
         @dragleave.prevent="dragover = false"
         :class="{ active: dragover }">
      <p v-if="!fileName">拖拽音频/视频文件到此处，或点击选择</p>
      <p v-else class="file-selected">{{ fileName }}</p>
      <input ref="fileInput" type="file" accept="audio/*,video/*" @change="handleFileSelect" hidden />
      <button @click="$refs.fileInput.click()" class="btn-choose">选择文件</button>
    </div>

    <!-- 处理控制 -->
    <div class="controls" v-if="fileName">
      <button @click="startTranscribe" :disabled="processing" class="btn-start">
        {{ processing ? '转写中...' : '开始转写' }}
      </button>
      <button @click="stopTranscribe" :disabled="!processing" class="btn-stop">停止</button>
      <button @click="clearFile" class="btn-clear">清除</button>
    </div>

    <!-- 进度 -->
    <div class="progress-bar" v-if="processing">
      <div class="progress-fill" :style="{ width: progress + '%' }"></div>
      <span>{{ progress }}%</span>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const emit = defineEmits(['transcription-result', 'status-change'])

const fileInput = ref(null)
const fileName = ref('')
const fileData = ref(null)
const dragover = ref(false)
const processing = ref(false)
const progress = ref(0)
const abortController = ref(null)

function handleFileSelect(e) {
  const file = e.target.files[0]
  if (!file) return
  loadFile(file)
}

function handleDrop(e) {
  dragover.value = false
  const file = e.dataTransfer.files[0]
  if (!file) return
  loadFile(file)
}

function loadFile(file) {
  fileName.value = file.name
  const reader = new FileReader()
  reader.onload = () => {
    fileData.value = reader.result
  }
  reader.readAsArrayBuffer(file)
}

function clearFile() {
  fileName.value = ''
  fileData.value = null
  progress.value = 0
  if (fileInput.value) fileInput.value.value = ''
}

/**
 * 使用 Web Audio API 将音频文件转换为 PCM 分块
 * 通过 WebSocket 发送到后端，后端返回实时转写结果
 */
async function startTranscribe() {
  if (!fileData.value || processing.value) return

  processing.value = true
  progress.value = 0
  abortController.value = new AbortController()

  emit('status-change', '正在连接...')

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${proto}//${location.host}/ws/transcribe`
  const socket = new WebSocket(wsUrl)
  socket.binaryType = 'arraybuffer'

  socket.onopen = async () => {
    emit('status-change', '正在处理音频...')

    try {
      // 使用 AudioContext 解码音频
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
      const audioBuffer = await audioCtx.decodeAudioData(fileData.value.slice(0))

      // 提取单声道 16kHz 音频数据
      const channelData = audioBuffer.getChannelData(0) // Float32Array

      // 分块发送: 每 1 秒一个 chunk
      const chunkDuration = 1.0 // 秒
      const chunkSize = Math.floor(16000 * chunkDuration)
      const totalChunks = Math.ceil(channelData.length / chunkSize)

      for (let i = 0; i < totalChunks; i++) {
        if (abortController.value?.signal.aborted) break

        const start = i * chunkSize
        const end = Math.min(start + chunkSize, channelData.length)
        const chunk = channelData.slice(start, end)

        // Float32 → Int16 PCM
        const int16 = new Int16Array(chunk.length)
        for (let j = 0; j < chunk.length; j++) {
          const s = Math.max(-1, Math.min(1, chunk[j]))
          int16[j] = s < 0 ? s * 0x8000 : s * 0x7FFF
        }

        socket.send(int16.buffer)
        progress.value = Math.round((end / channelData.length) * 100)

        // 小延迟，避免 flooding
        await new Promise(r => setTimeout(r, 50))
      }

      // 发送结束信号
      socket.send(JSON.stringify({ type: 'end' }))
      emit('status-change', '转写完成')

    } catch (e) {
      emit('status-change', '处理失败: ' + e.message)
      console.error(e)
    }
  }

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      if (data.type === 'transcription' && data.text) {
        emit('transcription-result', data)
      } else if (data.type === 'status') {
        emit('status-change', data.message)
      } else if (data.type === 'error') {
        emit('status-change', '错误: ' + data.message)
      }
    } catch {}
  }

  socket.onerror = () => {
    emit('status-change', 'WebSocket 连接失败，请确认后端已启动')
  }

  socket.onclose = () => {
    processing.value = false
  }
}

function stopTranscribe() {
  if (abortController.value) {
    abortController.value.abort()
  }
  processing.value = false
}
</script>

<style scoped>
.file-upload {
  margin-bottom: 16px;
}

.upload-zone {
  border: 2px dashed #45475a;
  border-radius: 12px;
  padding: 30px 20px;
  text-align: center;
  color: #a6adc8;
  transition: border-color 0.2s, background 0.2s;
  background: #1e1e2e;
}

.upload-zone.active {
  border-color: #89b4fa;
  background: #252540;
}

.upload-zone p {
  margin: 0 0 12px 0;
  font-size: 14px;
}

.file-selected {
  color: #89b4fa !important;
  font-weight: 600;
}

.btn-choose {
  padding: 8px 16px;
  background: #45475a;
  color: #cdd6f4;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
}

.btn-choose:hover {
  background: #585b70;
}

.controls {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

.controls button {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
}

.controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-start {
  background: #a6e3a1;
  color: #1e1e2e;
}

.btn-start:hover:not(:disabled) {
  background: #c3f0c1;
}

.btn-stop {
  background: #f38ba8;
  color: #1e1e2e;
}

.btn-clear {
  background: #45475a;
  color: #cdd6f4;
}

.progress-bar {
  margin-top: 12px;
  height: 24px;
  background: #313244;
  border-radius: 12px;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.progress-fill {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: linear-gradient(90deg, #89b4fa, #a6e3a1);
  transition: width 0.3s ease;
  border-radius: 12px;
}

.progress-bar span {
  position: relative;
  z-index: 1;
  font-size: 12px;
  font-weight: 600;
  color: #1e1e2e;
}
</style>
