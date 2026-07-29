import { defineConfig, type Plugin, type ViteUserConfig } from "vitest/config";
import { resolve } from "path";

const staticImageData: Plugin = {
  name: "next-static-image-data",
  enforce: "pre",
  load(id: string) {
    const path = id.split("?")[0];
    if (!/\.(svg|png|jpe?g|gif|webp|avif|ico|bmp)$/i.test(path)) return undefined;
    const name = path.slice(path.lastIndexOf("/") + 1);
    return `export default { src: ${JSON.stringify(`/_next/static/media/${name}`)}, height: 24, width: 24 };`;
  },
};

const config: ViteUserConfig = {
  plugins: [staticImageData],
  test: {
    environment: "jsdom",
    setupFiles: ["tests/setupTests.ts"],
    globals: true,
    css: true, // lets you import CSS/modules without extra mocks
    testTimeout: 30000,
    silent: process.env.CI ? "passed-only" : false,
    teardownTimeout: 60000,
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "**/*.d.ts",
        "**/*.test.*",
        "**/*.spec.*",

        "tests/**",

        "node_modules/**",
        ".next/**",
        "out/**",

        "**/*.config.*",
        "postcss.config.*",
        "tailwind.config.*",
        "next.config.*",
      ],
    },
    exclude: ["node_modules/**"],
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
};

export default defineConfig(config);
