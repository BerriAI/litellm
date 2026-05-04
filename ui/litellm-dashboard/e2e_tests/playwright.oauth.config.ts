/**
 * Minimal Playwright config for the MCP OAuth E2E test.
 * Runs against a pre-existing proxy on port 4000 with no globalSetup.
 */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: ["tests/mcp/mcp_oauth_flow.spec.ts"],
  fullyParallel: false,
  retries: 0,
  workers: 1,
  reporter: [["list"], ["html", { outputFolder: "playwright-report-oauth" }]],
  use: {
    baseURL: "http://localhost:4000",
    trace: "on-first-retry",
    actionTimeout: 20 * 1000,
    navigationTimeout: 45 * 1000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  timeout: 5 * 60 * 1000,
  expect: {
    timeout: 15 * 1000,
  },
  // No globalSetup — the test handles its own login
});
