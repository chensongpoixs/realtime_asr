import { createApp } from 'vue'
import App from './App.vue'
import VConsole from 'vconsole'

// 移动端调试工具（类似 PC 端 F12）
new VConsole()

createApp(App).mount('#app')
