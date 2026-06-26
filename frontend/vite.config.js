import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/ws': {
        target: 'wss://localhost:9765',
        ws: true,
      },
      '/api': {
        target: 'https://localhost:9765',
        changeOrigin: true,
      },
    },
  },
})
