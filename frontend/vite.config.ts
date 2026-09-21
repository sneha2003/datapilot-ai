import {defineConfig} from 'vite'; import react from '@vitejs/plugin-react';
const proxy={'/api':'http://127.0.0.1:8011'};
export default defineConfig({
  plugins:[react()],
  server:{port:5174,proxy},
  preview:{port:5174,proxy},
})
