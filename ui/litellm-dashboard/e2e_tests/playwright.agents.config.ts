/**
 * Playwright config for the cloud-agents suite.
 *
 * The agents UI lives in the Next.js App Router, which only renders under
 * `next dev` (not the proxy's static `output: "export"`). This config
 * targets http://localhost:3000 directly and skips the proxy globalSetup
 * that the rest of the suite needs.
 *
 * Run with:
 *   NEXT_PUBLIC_USE_MOCK_AGENTS=true pnpm dev
 *   pnpm exec playwright test --config e2e_tests/playwright.agents.config.ts
 */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/agents",
  testMatch: ["**/*.spec.ts"],
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "list",
  use: {
    baseURL: process.env.AGENTS_DEV_URL ?? "http://localhost:3000",
    trace: "on-first-retry",
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  timeout: 60_000,
  expect: { timeout: 10_000 },
});
