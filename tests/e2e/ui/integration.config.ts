import { defineConfig, devices } from "@playwright/test";
import * as path from "path";
import { ARTIFACT_DIR, UI_BASE_URL } from "./constants";

if (process.env.GITHUB_ACTIONS === "true")
  throw new Error("Integration contracts are owned by CircleCI");

export default defineConfig({
  testDir: "./tests/integrationCritical",
  testMatch: "*.spec.ts",
  fullyParallel: false,
  forbidOnly: true,
  retries: 0,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 10_000 },
  reporter: [
    ["line"],
    ["junit", { outputFile: path.join(ARTIFACT_DIR, "browser-junit.xml") }],
    ["json", { outputFile: path.join(ARTIFACT_DIR, "browser-results.json") }],
  ],
  outputDir: path.join(ARTIFACT_DIR, "browser-output"),
  use: {
    ...devices["Desktop Chrome"],
    baseURL: UI_BASE_URL,
    actionTimeout: 15_000,
    navigationTimeout: 30_000,
    trace: "retain-on-failure",
  },
});
