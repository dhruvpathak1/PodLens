import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const apiPort = process.env.API_PORT ?? '8000'

/** GitHub Pages project URL is /<repo>/; set VITE_BASE_PATH in CI (e.g. /PodLens_AI_Context_Generator/). */
function normalizeBase(raw: string | undefined): string {
  if (raw == null || String(raw).trim() === '' || String(raw).trim() === '/') return '/'
  let b = String(raw).trim()
  if (!b.startsWith('/')) b = `/${b}`
  if (!b.endsWith('/')) b = `${b}/`
  return b
}

// https://vite.dev/config/
export default defineConfig({
  base: normalizeBase(process.env.VITE_BASE_PATH),
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: `http://127.0.0.1:${apiPort}`,
        changeOrigin: true,
      },
    },
  },
})
