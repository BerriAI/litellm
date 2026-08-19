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

const sharedViteConfig = {
  plugins: [staticImageData],
  resolve: { alias: { "@": resolve(__dirname, "src") } },
  define: { "import.meta.vitest": "undefined" },
  esbuild: { jsx: "automatic", jsxImportSource: "react" } as const,
};

const TEST_TS_FILES_THAT_RENDER_REACT: readonly string[] = [
  "src/**/hooks/**/*.test.ts",
  "src/**/cost-tracking/_components/**/use_*.test.ts",
  "src/**/models-and-endpoints/detailNavigation.test.ts",
  "src/**/models-and-endpoints/vertexCredentialsUpload.test.ts",
  "src/components/chat/useChatHistory.test.ts",
  "src/lib/forms/pickDirty.test.ts",
];

const config: ViteUserConfig = {
  ...sharedViteConfig,
  test: {
    projects: [
      {
        ...sharedViteConfig,
        test: {
          name: "unit",
          environment: "node",
          globals: true,
          include: ["src/**/*.test.ts", "tests/**/*.test.ts"],
          exclude: ["node_modules/**", ...TEST_TS_FILES_THAT_RENDER_REACT],
          setupFiles: ["tests/setup.unit.ts"],
          testTimeout: 5_000,
          hookTimeout: 5_000,
        },
      },
      {
        ...sharedViteConfig,
        test: {
          name: "component",
          environment: "./tests/jsdomFetchEnv.ts",
          globals: true,
          css: true,
          include: ["src/**/*.test.tsx", "tests/**/*.test.tsx", ...TEST_TS_FILES_THAT_RENDER_REACT],
          exclude: ["node_modules/**", "**/*.integration.test.tsx"],
          setupFiles: ["tests/setupTests.ts"],
          testTimeout: 60_000,
          hookTimeout: 30_000,
        },
      },
      {
        ...sharedViteConfig,
        test: {
          name: "integration",
          environment: "./tests/jsdomFetchEnv.ts",
          globals: true,
          css: true,
          include: ["src/**/*.integration.test.tsx", "tests/**/*.integration.test.tsx"],
          exclude: ["node_modules/**"],
          setupFiles: ["tests/setupTests.ts"],
          testTimeout: 60_000,
          hookTimeout: 30_000,
        },
      },
      {
        ...sharedViteConfig,
        test: {
          name: "types",
          include: [],
          typecheck: {
            enabled: true,
            include: ["src/**/*.test-d.ts", "src/**/*.test-d.tsx"],
            ignoreSourceErrors: true,
          },
        },
      },
    ],
    silent: process.env.CI ? "passed-only" : false,
    retry: 0,
    teardownTimeout: 60_000,
    coverage: {
      provider: "v8",
      reporter: ["text", "lcov"],
      include: ["src/**/*.{ts,tsx}"],
      exclude: [
        "**/*.d.ts",
        "**/*.test.*",
        "**/*.test-d.*",
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
  },
};

export default defineConfig(config);
