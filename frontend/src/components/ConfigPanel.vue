<template>
  <div class="config-panel">
    <h3>模型配置</h3>
    <div class="config-grid">
      <!-- 模型来源切换 -->
      <label class="full-width">
        模型来源
        <div class="toggle-group">
          <button
            :class="['toggle-btn', { active: modelSource === 'preset' }]"
            @click="modelSource = 'preset'"
          >预设模型</button>
          <button
            :class="['toggle-btn', { active: modelSource === 'local' }]"
            @click="modelSource = 'local'"
          >本地路径</button>
        </div>
      </label>

      <!-- 预设模型下拉 -->
      <label v-if="modelSource === 'preset'">
        模型名称
        <select v-model="form.model_path">
          <option value="tiny">tiny</option>
          <option value="tiny.en">tiny.en (仅英文)</option>
          <option value="base">base</option>
          <option value="base.en">base.en (仅英文)</option>
          <option value="small">small</option>
          <option value="small.en">small.en (仅英文)</option>
          <option value="medium">medium</option>
          <option value="medium.en">medium.en (仅英文)</option>
          <option value="large-v2">large-v2</option>
          <option value="large-v3">large-v3</option>
        </select>
      </label>

      <!-- 本地模型路径输入 -->
      <label v-if="modelSource === 'local'" class="full-width">
        本地模型路径
        <input
          type="text"
          v-model="form.model_path"
          placeholder="/home/user/models/faster-whisper-large-v3"
        />
      </label>

      <!-- 模型下载目录（仅预设模型时显示） -->
      <label v-if="modelSource === 'preset'">
        下载目录
        <input type="text" v-model="form.download_root" placeholder="./models" />
      </label>

      <!-- HuggingFace 镜像（仅预设模型时显示） -->
      <label v-if="modelSource === 'preset'">
        HF 镜像
        <input type="text" v-model="form.hf_endpoint" placeholder="https://hf-mirror.com" />
      </label>

      <label>
        设备
        <select v-model="form.device">
          <option value="cpu">CPU</option>
          <option value="cuda">CUDA (GPU)</option>
        </select>
      </label>

      <label>
        计算精度
        <select v-model="form.compute_type">
          <option value="int8">int8</option>
          <option value="int8_float16">int8_float16</option>
          <option value="float16">float16</option>
        </select>
      </label>

      <label>
        语言
        <select v-model="form.language">
          <option value="auto">自动检测</option>
          <option value="zh">中文</option>
          <option value="en">英文</option>
          <option value="ja">日文</option>
          <option value="ko">韩文</option>
        </select>
      </label>

      <label>
        缓冲阈值（秒）
        <input type="number" v-model.number="form.buffer_threshold" min="0.5" max="10" step="0.5" />
      </label>

      <label>
        后端地址
        <input type="text" v-model="form.backend_url" placeholder="http://localhost:8765" />
      </label>
    </div>

    <div class="config-actions">
      <button @click="saveConfig" :disabled="saving">
        {{ saving ? '保存中...' : '保存配置' }}
      </button>
      <button @click="loadConfig" class="btn-secondary">刷新配置</button>
    </div>

    <div v-if="statusMsg" class="config-status">{{ statusMsg }}</div>
  </div>
</template>

<script setup>
import { reactive, ref, onMounted } from 'vue'

const emit = defineEmits(['configChanged'])

const modelSource = ref('preset')

const form = reactive({
  model_path: 'medium',
  download_root: './models',
  hf_endpoint: 'https://hf-mirror.com',
  device: 'cpu',
  compute_type: 'int8',
  language: 'auto',
  buffer_threshold: 2.0,
  backend_url: '',
})

const saving = ref(false)
const statusMsg = ref('')

function api(path, opts = {}) {
  const base = form.backend_url ? form.backend_url.replace(/\/$/, '') : ''
  return fetch(`${base}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
}

async function loadConfig() {
  try {
    const res = await api('/api/config')
    if (res.ok) {
      const data = await res.json()
      if (data.model) {
        const mp = data.model.model_path || ''
        // 判断是预设模型还是本地路径
        const presets = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v2', 'large-v3']
        if (presets.includes(mp)) {
          modelSource.value = 'preset'
          form.model_path = mp
        } else if (mp) {
          modelSource.value = 'local'
          form.model_path = mp
        }
        form.download_root = data.model.download_root || './models'
        form.hf_endpoint = data.model.hf_endpoint || 'https://hf-mirror.com'
        form.device = data.model.device || form.device
        form.compute_type = data.model.compute_type || form.compute_type
      }
      if (data.transcription) {
        form.language = data.transcription.language || form.language
        form.buffer_threshold = data.transcription.buffer_threshold || form.buffer_threshold
      }
      statusMsg.value = '配置已加载'
    }
  } catch (e) {
    statusMsg.value = '加载失败: ' + e.message
  }
}

async function saveConfig() {
  saving.value = true
  statusMsg.value = ''
  try {
    const res = await api('/api/config', {
      method: 'POST',
      body: JSON.stringify({
        model: {
          model_path: form.model_path,
          download_root: form.download_root,
          hf_endpoint: form.hf_endpoint,
          device: form.device,
          compute_type: form.compute_type,
        },
        transcription: {
          language: form.language,
          buffer_threshold: form.buffer_threshold,
        },
      }),
    })
    if (res.ok) {
      statusMsg.value = '配置已保存，模型已重新加载'
      emit('configChanged', { ...form })
    } else {
      statusMsg.value = '保存失败: ' + (await res.text())
    }
  } catch (e) {
    statusMsg.value = '保存失败: ' + e.message
  } finally {
    saving.value = false
  }
}

onMounted(loadConfig)
</script>

<style scoped>
.config-panel {
  background: #1e1e2e;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
}

.config-panel h3 {
  margin: 0 0 16px 0;
  color: #cdd6f4;
  font-size: 16px;
}

.config-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 12px;
}

.config-grid label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 13px;
  color: #a6adc8;
}

.config-grid label.full-width {
  grid-column: 1 / -1;
}

.config-grid select,
.config-grid input {
  padding: 8px 10px;
  border: 1px solid #45475a;
  border-radius: 6px;
  background: #313244;
  color: #cdd6f4;
  font-size: 14px;
}

.config-grid select:focus,
.config-grid input:focus {
  outline: none;
  border-color: #89b4fa;
}

.toggle-group {
  display: flex;
  gap: 0;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid #45475a;
}

.toggle-btn {
  flex: 1;
  padding: 8px 12px;
  border: none;
  border-radius: 0;
  background: #313244;
  color: #a6adc8;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.2s, color 0.2s;
}

.toggle-btn.active {
  background: #89b4fa;
  color: #1e1e2e;
  font-weight: 600;
}

.toggle-btn:not(.active):hover {
  background: #45475a;
}

.config-actions {
  display: flex;
  gap: 8px;
  margin-top: 16px;
}

button {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background: #89b4fa;
  color: #1e1e2e;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.2s;
}

button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

button:hover:not(:disabled) {
  background: #b4d0fb;
}

.btn-secondary {
  background: #45475a;
  color: #cdd6f4;
}

.btn-secondary:hover {
  background: #585b70;
}

.config-status {
  margin-top: 12px;
  padding: 8px 12px;
  background: #313244;
  border-radius: 6px;
  font-size: 13px;
  color: #a6e3a1;
}
</style>
