import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies /api to the Flask backend, so the browser sees ONE origin and no
// CORS is involved during development. In a real deployment the built assets would be
// served by (or next to) Flask directly.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { '/api': 'http://127.0.0.1:5000' },
  },
})
