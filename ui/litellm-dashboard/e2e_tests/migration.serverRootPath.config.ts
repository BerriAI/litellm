import { defineConfig, devices } from "@playwright/test";

/**
 * App Router migration smoke under a non-root mount. Boot the proxy with the same
 * SERVER_ROOT_PATH (e.g. SERVER_ROOT_PATH=/litellm) and a UI built for it before
 * running. globalSetup logs in at `${SERVER_ROOT_PATH}/ui/login` so the admin
 * storage state is valid under the prefix.
 */
if (!process.env.SERVER_ROOT_PATH) {
  throw new Error(
    "migration.serverRootPath.config.ts requires SERVER_ROOT_PATH to be set (e.g. SERVER_ROOT_PATH=/litellm). " +
      "Without it this config silently re-runs the default mount and never exercises the prefix. " +
      "For the root-less run use the default playwright.config.ts (npm run e2e:migration).",
  );
}

export default defineConfig({
  testDir: "./tests/migration",
  testMatch: ["migratedPages.spec.ts"],
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "list",
  use: {
    baseURL: "http://localhost:4000",
    trace: "on-first-retry",
    actionTimeout: 15 * 1000,
    navigationTimeout: 30 * 1000,
    launchOptions: {
      slowMo: process.env.SLOWMO ? parseInt(process.env.SLOWMO, 10) || 0 : 0,
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  timeout: 3 * 60 * 1000,
  expect: { timeout: 10 * 1000 },
  globalSetup: require.resolve("./globalSetup"),
});
