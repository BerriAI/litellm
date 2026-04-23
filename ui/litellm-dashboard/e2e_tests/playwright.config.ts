import { defineConfig, devices } from "@playwright/test";

/**
 * See https://playwright.dev/docs/test-configuration.
 *
 * Two projects:
 *   - `chromium`: existing full e2e suite against the proxy at :4000.
 *   - `parity`:   phase-1 shadcn migration parity specs under `./parity/`.
 *                 Uses a dev-server baseURL (3000) by default but can be
 *                 overridden via `PARITY_BASE_URL` (e.g. to aim at the
 *                 proxy at 4000 if dev-server auth mocking proves too
 *                 brittle). The webServer entry starts `npm run dev`
 *                 automatically if PARITY_BASE_URL is not set.
 */
const PARITY_BASE_URL = process.env.PARITY_BASE_URL ?? "http://localhost:3000";
const PARITY_USES_LOCAL_DEV = !process.env.PARITY_BASE_URL;

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
  reporter: "html",
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: "http://localhost:4000",

    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: "on-first-retry",

    /* Action timeout for clicks, fills, waitForSelector, etc. */
    actionTimeout: 15 * 1000,
    navigationTimeout: 30 * 1000,
  },

  /* Configure projects for major browsers */
  projects: [
    {
      name: "chromium",
      testDir: ".",
      testIgnore: ["parity/**", "**/*.test.*"],
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "parity",
      testDir: "./parity",
      testMatch: ["**/*.spec.ts"],
      use: {
        ...devices["Desktop Chrome"],
        baseURL: PARITY_BASE_URL,
        // Lock viewport for stable visual snapshots across environments.
        viewport: { width: 1440, height: 900 },
      },
    },
  ],

  /* Auto-start the Next dev server for parity specs unless PARITY_BASE_URL
     points elsewhere (e.g. the proxy at :4000 for auth-backed parity runs). */
  webServer: PARITY_USES_LOCAL_DEV
    ? {
        command: "npm run dev",
        cwd: "..",
        url: "http://localhost:3000",
        reuseExistingServer: true,
        timeout: 120 * 1000,
      }
    : undefined,

  /* Timeout settings */
  timeout: 3 * 60 * 1000,
  expect: {
    timeout: 10 * 1000,
    /* Keep visual snapshot diffs tight so real regressions aren't masked. */
    toHaveScreenshot: {
      maxDiffPixelRatio: 0.01,
      animations: "disabled",
      caret: "hide",
    },
  },
  globalSetup: require.resolve("./globalSetup"),
});
