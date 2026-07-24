import { defineConfig, devices } from "@playwright/test";

// Minimal config for the SERVER_ROOT_PATH redirect spec. Deliberately does NOT
// reuse the main e2e config because:
//   - globalSetup logs in via http://localhost:4000/ui/login, which 404s when
//     the proxy is mounted under a non-root path.
//   - The redirect spec must run against a clean, unauthenticated session, so
//     no storage state should be loaded.
export default defineConfig({
  testDir: "./tests/login",
  testMatch: ["serverRootPathRedirect.spec.ts"],
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: 1,
  reporter: "list",
  use: {
    trace: "on-first-retry",
    actionTimeout: 15 * 1000,
    navigationTimeout: 30 * 1000,
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  timeout: 60 * 1000,
  expect: {
    timeout: 10 * 1000,
  },
});
