/**
 * Minimal Playwright config for running just the fixture liveness meta-test
 * in isolation — no proxy, no DB, no globalSetup. Used by CI and locally to
 * sanity-check that the guarded-page fixture wires up correctly.
 */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: ".",
  testMatch: ["tests/meta/*.spec.ts"],
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: [["list"]],
  use: {
    actionTimeout: 5 * 1000,
    navigationTimeout: 10 * 1000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  timeout: 30 * 1000,
  expect: { timeout: 5 * 1000 },
});
