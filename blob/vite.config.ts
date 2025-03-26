import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
// @ts-ignore
import civet from '@danielx/civet/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [civet({}), react()],
})
