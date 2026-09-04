import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env files
  const env = loadEnv(mode, process.cwd(), '');
  const target = env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

  return {
    plugins: [
      react(),
      tailwindcss(),
    ],
    server: {
      proxy: {
        '/predict': {
          target: target,
          changeOrigin: true,
        },
        // Commentary, split out from /predict (see phase6b_api.py and
        // App.jsx's triggerPrediction) so a slow/rate-limited Groq call
        // never blocks the win-probability response. Needs its own dev-
        // proxy entry for the same reason /predict does — without this,
        // `npm run dev` would send /explain to the Vite dev server
        // itself (which has no such route) instead of the FastAPI
        // backend. Production is unaffected: VITE_API_BASE_URL there
        // points requests straight at the deployed backend origin, no
        // proxy involved.
        '/explain': {
          target: target,
          changeOrigin: true,
        },
        '/health_check': {
          target: target,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/health_check/, ''),
        }
      }
    }
  }
})