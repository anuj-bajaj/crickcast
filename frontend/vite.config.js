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
        '/health_check': {
          target: target,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/health_check/, ''),
        }
      }
    }
  }
})
