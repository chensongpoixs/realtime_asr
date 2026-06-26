<template>
  <div class="file-upload">
    <!-- 模式切换 -->
    <div class="mode-tabs">
      <button
        :class="['mode-tab', { active: mode === 'file' }]"
        @click="switchMode('file')"
      >文件转写</button>
      <button
        :class="['mode-tab', { active: mode === 'mic' }]"
        @click="switchMode('mic')"
      >麦克风录音</button>
    </div>

    <!-- ═══ 文件上传模式 ═══ -->
    <div v-if="mode === 'file'">
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

      <div class="controls" v-if="fileName">
        <button @click="startFileTranscribe" :disabled="processing" class="btn-start">
          {{ processing ? '转写中...' : '开始转写' }}
        </button>
        <button @click="stopTranscribe" :disabled="!processing" class="btn-stop">停止</button>
        <button @click="clearFile" class="btn-clear">清除</button>
      </div>

      <div class="progress-bar" v-if="processing && mode === 'file'">
        <div class="progress-fill" :style="{ width: progress + '%' }"></div>
        <span>{{ progress }}%</span>
      </div>
    </div>

    <!-- ═══ 麦克风录音模式 ═══ -->
    <div v-if="mode === 'mic'" class="mic-section">

      <div class="mic-visual" :class="{ recording: micRecording }">
        <div class="mic-icon">{{ micRecording ? '🎙️' : '🎤' }}</div>
        <div class="mic-status">{{ micStatusText }}</div>
        <div class="recording-indicator" v-if="micRecording">
          <span class="pulse"></span>
          录音中 {{ formatDuration(micDuration) }}
        </div>
      </div>

      <div class="controls">
        <button
          v-if="!micRecording"
          @click="startMicRecord"
          :disabled="processing"
          class="btn-start"
        >开始录音</button>
        <button
          v-else
          @click="stopMicRecord"
          class="btn-stop"
        >停止录音</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onUnmounted } from 'vue'

const emit = defineEmits(['transcription-result', 'status-change'])

// ─── 模式切换 ───
const mode = ref('file')

function switchMode(m) {
  if (processing.value) return // 处理中不允许切换
  mode.value = m
}

// ─── 文件模式 ───
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
  reader.onload = () => { fileData.value = reader.result }
  reader.readAsArrayBuffer(file)
}

function clearFile() {
  fileName.value = ''
  fileData.value = null
  progress.value = 0
  if (fileInput.value) fileInput.value.value = ''
}

function createWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${location.host}/ws/transcribe`
  console.log('[WS] 创建连接:', url)
  return new WebSocket(url)
}

function float32ToInt16(float32) {
  const int16 = new Int16Array(float32.length)
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]))
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
  }
  return int16
}

/** 简单的线性插值重采样 */
function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return new Float32Array(samples)
  const ratio = fromRate / toRate
  const outLen = Math.floor(samples.length / ratio)
  const out = new Float32Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio
    const srcFloor = Math.floor(srcIdx)
    const srcCeil = Math.min(srcFloor + 1, samples.length - 1)
    const t = srcIdx - srcFloor
    out[i] = samples[srcFloor] * (1 - t) + samples[srcCeil] * t
  }
  return out
}

async function startFileTranscribe() {
  if (!fileData.value || processing.value) return
  processing.value = true
  progress.value = 0
  abortController.value = new AbortController()
  emit('status-change', '正在连接...')

  const socket = createWebSocket()
  socket.binaryType = 'arraybuffer'

  socket.onopen = async () => {
    console.log('[WS-file] 连接已建立，开始发送音频')
    emit('status-change', '正在处理音频...')
    try {
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 })
      const audioBuffer = await audioCtx.decodeAudioData(fileData.value.slice(0))
      const channelData = audioBuffer.getChannelData(0)
      console.log('[WS-file] 音频解码完成, samples=', channelData.length, ', duration=', (channelData.length / 16000).toFixed(1) + 's')

      const chunkDuration = 1.0
      const chunkSize = Math.floor(16000 * chunkDuration)
      const totalChunks = Math.ceil(channelData.length / chunkSize)
      console.log('[WS-file] 分块发送, chunkSize=', chunkSize, ', totalChunks=', totalChunks)

      for (let i = 0; i < totalChunks; i++) {
        if (abortController.value?.signal.aborted) break
        const start = i * chunkSize
        const end = Math.min(start + chunkSize, channelData.length)
        const int16 = float32ToInt16(channelData.slice(start, end))
        socket.send(int16.buffer)
        console.log(`[WS-file] 发送chunk ${i + 1}/${totalChunks}, bytes=${int16.byteLength}`)
        progress.value = Math.round((end / channelData.length) * 100)
        await new Promise(r => setTimeout(r, 50))
      }

      socket.send(JSON.stringify({ type: 'end' }))
      console.log('[WS-file] 所有块发送完毕，已发送 end 标记')
      emit('status-change', '转写完成')
    } catch (e) {
      console.error('[WS-file] 处理失败:', e)
      emit('status-change', '处理失败: ' + e.message)
    }
  }

  setupSocketHandlers(socket, 'WS-file')
}

function setupSocketHandlers(socket, label = 'WS') {
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      console.log(`[${label}] 收到消息:`, data.type, data.text ? `text.length=${data.text.length}` : '', data)
      if (data.type === 'transcription' && data.text) {
        emit('transcription-result', data)
      } else if (data.type === 'status') {
        emit('status-change', data.message)
      } else if (data.type === 'error') {
        emit('status-change', '错误: ' + data.message)
      }
    } catch {
      console.log(`[${label}] 收到非JSON消息, bytes=${event.data instanceof ArrayBuffer ? event.data.byteLength : typeof event.data}`)
    }
  }
  socket.onerror = (e) => {
    console.error(`[${label}] 连接错误, readyState=${socket.readyState}, url=${socket.url}`, e)
    emit('status-change', 'WebSocket 连接失败，请确认后端已启动')
  }
  socket.onclose = (e) => {
    console.log(`[${label}] 连接关闭, code=${e.code}, reason="${e.reason}", wasClean=${e.wasClean}`)
    processing.value = false
  }
}

function stopTranscribe() {
  if (abortController.value) abortController.value.abort()
  processing.value = false
}

// ─── 麦克风模式 ───
const micRecording = ref(false)
const micStatusText = ref('点击开始录音')
const micDuration = ref(0)
const isIOS = /iPhone|iPad|iPod/.test(navigator.userAgent) ||
  (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1)
let micStream = null
let micAudioCtx = null
let micProcessor = null
let micSocket = null
let micTimer = null
let micBuffer = []       // Float32 样本累积
let micChunkMs = 2000    // 每2秒发送一次
const micSampleRate = 16000

async function startMicRecord() {
  if (processing.value) return
  processing.value = true
  micBuffer = []

  try {
    // 1. 先获取麦克风权限（必须在用户手势回调中同步调用，手机端尤其重要）
    emit('status-change', '正在请求麦克风权限...')
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: { ideal: micSampleRate },
        channelCount: { ideal: 1 },
        echoCancellation: true,
        noiseSuppression: true,
      }
    })
      //micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
     // audioTrack = micStream.getAudioTracks()[0];
     // settings = audioTrack.getSettings();
     // micSampleRate = settings.sampleRate; // 这就是设备当前的采样率，如 48000
    console.log(`麦克风实际采样率: ${micSampleRate}`);

    // 2. 连接 WebSocket
    emit('status-change', '正在连接服务器...')
    micSocket = createWebSocket()
    micSocket.binaryType = 'arraybuffer'

    await new Promise((resolve, reject) => {
      micSocket.onopen = () => {
        console.log('[WS-mic] 连接已建立, readyState=', micSocket.readyState)
        resolve()
      }
      micSocket.onerror = () => {
        console.error('[WS-mic] 连接错误, readyState=', micSocket.readyState, ', url=', micSocket.url)
        reject(new Error('WebSocket连接失败，请确认后端已启动'))
      }
      setTimeout(() => {
        console.error('[WS-mic] 连接超时 (8s)')
        reject(new Error('连接超时，请检查网络'))
      }, 8000)
    })

    setupSocketHandlers(micSocket, 'WS-mic')

    // 3. 创建 AudioContext（手机端必须用用户手势恢复挂起状态）
    micAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: micSampleRate })
    console.log('[Mic] AudioContext 初始状态:', micAudioCtx.state, ', sampleRate:', micAudioCtx.sampleRate)
    if (micAudioCtx.state === 'suspended') {
      await micAudioCtx.resume()
      console.log('[Mic] AudioContext 已恢复, state=', micAudioCtx.state)
    }
    const source = micAudioCtx.createMediaStreamSource(micStream)

    // 获取实际采样率（手机端可能与理想值不同，需要适配）
    const actualSampleRate = micAudioCtx.sampleRate
    // bufferSize 根据实际采样率调整：手机上保持约 256ms 的缓冲区
    const bufferSize = Math.pow(2, Math.ceil(Math.log2(actualSampleRate * 0.25)))
    console.log('[Mic] 实际采样率:', actualSampleRate, ', 目标:', micSampleRate, ', bufferSize:', bufferSize)

    // 使用 ScriptProcessorNode 捕获原始 PCM
    micProcessor = micAudioCtx.createScriptProcessor(bufferSize, 1, 1)

    const sendInterval = actualSampleRate * (micChunkMs / 1000)
    console.log('[Mic] sendInterval=', sendInterval, ', chunkDuration=', micChunkMs + 'ms')
    let micChunkCount = 0
    let micBytesSent = 0

    micProcessor.onaudioprocess = (event) => {
      if (!micRecording.value) return
      const inputData = event.inputBuffer.getChannelData(0) // Float32Array
      micBuffer.push(...inputData)

      // 降采样到 16000Hz（如果实际采样率不同）
      while (micBuffer.length >= sendInterval) {
        const chunk = micBuffer.splice(0, sendInterval)
        // 重采样到 16kHz
        const resampled = actualSampleRate !== micSampleRate
          ? resample(chunk, actualSampleRate, micSampleRate)
          : new Float32Array(chunk)
        const int16 = float32ToInt16(resampled)
        if (micSocket && micSocket.readyState === WebSocket.OPEN) {
          micSocket.send(int16.buffer)
          micChunkCount++
          micBytesSent += int16.byteLength
          console.log(`[Mic] 发送chunk #${micChunkCount}, bytes=${int16.byteLength}, 累计=${(micBytesSent / 1024).toFixed(0)}KB`)
        }
      }
    }

    source.connect(micProcessor)
    // 必须连接到 destination，否则 onaudioprocess 不会触发
    micProcessor.connect(micAudioCtx.destination)

    // 4. 开始
    micChunkCount = 0
    micBytesSent = 0
    micRecording.value = true
    micDuration.value = 0
    micStatusText.value = '录音中...'
    emit('status-change', '录音中 (采样率: ' + actualSampleRate + 'Hz)')

    micTimer = setInterval(() => { micDuration.value++ }, 1000)

  } catch (e) {
    processing.value = false
    micRecording.value = false
    const msg = e.name === 'NotAllowedError'
      ? '麦克风权限被拒绝，请在浏览器设置中允许麦克风访问'
      : e.name === 'NotFoundError'
      ? '未检测到麦克风设备'
      : '麦克风启动失败: ' + e.message
    emit('status-change', msg)
    console.error('Mic error:', e)
    cleanupMic()
  }
}

function stopMicRecord() {
  console.log('[Mic] 停止录音, 剩余buffer=', micBuffer.length)
  micRecording.value = false
  micStatusText.value = '正在处理剩余音频...'
  emit('status-change', '正在处理剩余音频...')

  // 发送剩余缓冲数据
  if (micBuffer.length > 0 && micSocket && micSocket.readyState === WebSocket.OPEN) {
    const int16 = float32ToInt16(new Float32Array(micBuffer))
    micSocket.send(int16.buffer)
    console.log('[Mic] 发送剩余数据, bytes=', int16.byteLength)
    micBuffer = []
  }

  // 发送结束信号
  if (micSocket && micSocket.readyState === WebSocket.OPEN) {
    micSocket.send(JSON.stringify({ type: 'end' }))
    console.log('[Mic] 已发送 end 标记')
  }

  micStatusText.value = '点击开始录音'
  emit('status-change', '转写完成')
  processing.value = false
  cleanupMic()
}

function cleanupMic() {
  console.log('[Mic] 清理资源...')
  if (micTimer) { clearInterval(micTimer); micTimer = null }
  if (micProcessor) {
    micProcessor.disconnect()
    micProcessor = null
  }
  if (micAudioCtx) {
    micAudioCtx.close()
    micAudioCtx = null
  }
  if (micStream) {
    micStream.getTracks().forEach(t => t.stop())
    micStream = null
  }
  // WebSocket 延迟关闭，等后端处理完
  setTimeout(() => {
    if (micSocket) { micSocket.close(); micSocket = null }
  }, 500)
}

function formatDuration(sec) {
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

onUnmounted(() => {
  stopTranscribe()
  if (micRecording.value) stopMicRecord()
  cleanupMic()
})
</script>

<style scoped>
.file-upload {
  margin-bottom: 16px;
}

/* ─── 模式切换标签 ─── */
.mode-tabs {
  display: flex;
  gap: 0;
  margin-bottom: 16px;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #d0d0d0;
}

.mode-tab {
  flex: 1;
  padding: 10px 16px;
  border: none;
  border-radius: 0;
  background: #f5f5f5;
  color: #666;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s, color 0.2s;
}

.mode-tab.active {
  background: #4a6cf7;
  color: #fff;
}

.mode-tab:not(.active):hover {
  background: #e8e8e8;
}

/* ─── 文件上传区域 ─── */
.upload-zone {
  border: 2px dashed #d0d0d0;
  border-radius: 12px;
  padding: 30px 20px;
  text-align: center;
  color: #999;
  transition: border-color 0.2s, background 0.2s;
  background: #fff;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

.upload-zone.active {
  border-color: #4a6cf7;
  background: #f0f3ff;
}

.upload-zone p {
  margin: 0 0 12px 0;
  font-size: 14px;
}

.file-selected {
  color: #4a6cf7 !important;
  font-weight: 600;
}

.btn-choose {
  padding: 8px 16px;
  background: #e8e8e8;
  color: #333;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  cursor: pointer;
}

.btn-choose:hover { background: #d5d5d5; }

/* ─── 麦克风区域 ─── */
.mic-section {
  background: #fff;
  border-radius: 12px;
  padding: 24px 20px;
  text-align: center;
  box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

.mic-visual {
  margin-bottom: 16px;
}

.mic-icon {
  font-size: 48px;
  margin-bottom: 8px;
  transition: transform 0.3s;
}

.mic-visual.recording .mic-icon {
  animation: micPulse 1.5s ease-in-out infinite;
}

@keyframes micPulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.15); }
}

.mic-status {
  color: #888;
  font-size: 14px;
  margin-bottom: 8px;
}

.recording-indicator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: #e53935;
  font-size: 14px;
  font-weight: 600;
}

.pulse {
  width: 10px;
  height: 10px;
  background: #e53935;
  border-radius: 50%;
  animation: pulse 1s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.7); }
}

/* ─── 控制按钮 ─── */
.controls {
  display: flex;
  gap: 8px;
  margin-top: 12px;
  justify-content: center;
  flex-wrap: wrap;
}

.controls button {
  padding: 10px 22px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s, background 0.2s;
  white-space: nowrap;
}

.controls button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-start {
  background: #4caf50;
  color: #fff;
}

.btn-start:hover:not(:disabled) { background: #43a047; }

.btn-stop {
  background: #e53935;
  color: #fff;
}

.btn-stop:hover:not(:disabled) { background: #d32f2f; }

.btn-clear {
  background: #e8e8e8;
  color: #333;
}

.btn-clear:hover { background: #d5d5d5; }

/* ─── 进度条 ─── */
.progress-bar {
  margin-top: 12px;
  height: 24px;
  background: #eee;
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
  background: linear-gradient(90deg, #4a6cf7, #66bb6a);
  transition: width 0.3s ease;
  border-radius: 12px;
}

.progress-bar span {
  position: relative;
  z-index: 1;
  font-size: 12px;
  font-weight: 600;
  color: #fff;
}

/* ─── 手机端 ─── */
@media (max-width: 768px) {
  .upload-zone {
    padding: 20px 14px;
  }
  .mic-section {
    padding: 18px 14px;
  }
  .mic-icon {
    font-size: 40px;
  }
  .controls button {
    padding: 10px 18px;
    font-size: 13px;
  }
}
</style>
