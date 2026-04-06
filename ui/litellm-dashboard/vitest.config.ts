import { defineConfig } from "vitest/config";
import { resolve } from "path";

export default defineConfig({
  test: {
    environment: "jsdom",
    setupFiles: ["tests/setupTests.ts"],
    globals: true,
    css: true, // lets you import CSS/modules without extra mocks
    testTimeout: 10000,
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "**/*.d.ts",
        "**/*.test.*",
        "**/*.spec.*",

        "tests/**",
        "e2e_tests/**",

        "node_modules/**",
        ".next/**",
        "out/**",

        "**/*.config.*",
        "postcss.config.*",
        "tailwind.config.*",
        "next.config.*",
      ],
    },
    exclude: ["e2e_tests/**", "node_modules/**"],
    include: ["src/**/*.test.ts", "src/**/*.test.tsx", "tests/**/*.test.ts", "tests/**/*.test.tsx"],
  },
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  define: {
    "import.meta.vitest": "undefined",
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "react",
  },
});
