import { defineConfig, devices } from "@playwright/test";

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: ".",
  testMatch: ["**/*.spec.ts", "**/*.setup.ts"],
  testIgnore: ["**/*.test.*"],
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* One worker everywhere: several specs mutate global gateway state
     (router_settings, public model groups), so parallel workers race each
     other against the single shared gateway. */
  workers: 1,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: "html",
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: process.env.LITELLM_PROXY_URL ?? "http://localhost:4000",

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: "on-first-retry",

    /* Action timeout for clicks, fills, waitForSelector, etc. */
    actionTimeout: 15 * 1000,
    navigationTimeout: 30 * 1000,

    /* Slow down actions when SLOWMO=<ms> is set, useful for headed local debugging */
    launchOptions: {
      slowMo: process.env.SLOWMO ? parseInt(process.env.SLOWMO, 10) || 0 : 0,
    },
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  /* Timeout settings */
  timeout: 3 * 60 * 1000,
  expect: {
    timeout: 10 * 1000,
  },
  globalSetup: require.resolve("./globalSetup"),
});
