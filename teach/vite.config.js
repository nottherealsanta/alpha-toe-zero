import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        mcts: resolve(__dirname, 'pages/mcts.html'),
        'nn': resolve(__dirname, 'pages/nn.html'),
        'self-play': resolve(__dirname, 'pages/self-play.html'),
      },
    },
  },
})
