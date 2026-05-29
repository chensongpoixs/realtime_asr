/**
 * WebSocket 连接管理 composable
 * 负责连接生命周期、消息收发、自动重连
 */
import { ref, shallowRef, onUnmounted } from 'vue'

export function useWebSocket(url) {
  const connected = ref(false)
  const messages = ref([])
  const error = ref(null)
  const ws = shallowRef(null)
  let reconnectTimer = null
  let reconnectAttempts = 0
  const maxReconnectAttempts = 5
  const reconnectDelay = 2000

  function connect() {
    if (ws.value && (ws.value.readyState === WebSocket.OPEN || ws.value.readyState === WebSocket.CONNECTING)) {
      return
    }

    const socket = new WebSocket(url)
    socket.binaryType = 'arraybuffer'

    socket.onopen = () => {
      connected.value = true
      error.value = null
      reconnectAttempts = 0
      console.log('[WS] 已连接')
    }

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        messages.value = [...messages.value, data]
      } catch {
        console.warn('[WS] 无法解析消息:', event.data)
      }
    }

    socket.onerror = (e) => {
      error.value = 'WebSocket 连接错误'
      console.error('[WS] 错误:', e)
    }

    socket.onclose = (e) => {
      connected.value = false
      ws.value = null
      console.log('[WS] 已断开, code:', e.code)

      if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++
        console.log(`[WS] ${reconnectDelay / 1000}s 后重连 (${reconnectAttempts}/${maxReconnectAttempts})`)
        reconnectTimer = setTimeout(connect, reconnectDelay)
      }
    }

    ws.value = socket
  }

  function send(data) {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      ws.value.send(data)
    }
  }

  function sendBinary(buffer) {
    if (ws.value && ws.value.readyState === WebSocket.OPEN) {
      ws.value.send(buffer)
    }
  }

  function sendJson(obj) {
    send(JSON.stringify(obj))
  }

  function disconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
    reconnectAttempts = maxReconnectAttempts // 阻止重连
    if (ws.value) {
      ws.value.close()
      ws.value = null
    }
  }

  function clearMessages() {
    messages.value = []
  }

  onUnmounted(() => {
    disconnect()
  })

  return {
    connected,
    messages,
    error,
    connect,
    disconnect,
    send,
    sendBinary,
    sendJson,
    clearMessages,
  }
}
