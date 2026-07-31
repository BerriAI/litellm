import { defineConfig, devices } from "@playwright/test";
import * as path from "path";
import { ARTIFACT_DIR, UI_BASE_URL } from "./constants";

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
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 1 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  /* The html reporter and the artifact dir both write relative to cwd, which is
     read-only in the packaged e2e image; keep them under ARTIFACT_DIR so a plain
     `npx playwright test` works there without extra flags. */
  reporter: [["html", { outputFolder: path.join(ARTIFACT_DIR, "playwright-report"), open: "never" }]],
  outputDir: path.join(ARTIFACT_DIR, "test-results"),
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: UI_BASE_URL,

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
