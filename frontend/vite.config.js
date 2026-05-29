import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      '/ws': {
        target: 'ws://localhost:9765',
        ws: true,
      },
      '/api': {
        target: 'http://localhost:9765',
        changeOrigin: true,
      },
    },
  },
})
