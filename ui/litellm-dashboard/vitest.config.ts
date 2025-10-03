import { defineConfig } from 'vitest/config'
import { resolve } from "path"

export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['tests/setupTests.ts'],
    globals: true,
    css: true, // lets you import CSS/modules without extra mocks
    coverage: { reporter: ['text', 'lcov'] },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
})
