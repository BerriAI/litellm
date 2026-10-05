import { defineConfig, devices } from "@playwright/test";
import * as path from "path";

const baseURL = process.env.E2E_OIDC_UI_URL;
if (!baseURL) throw new Error("E2E_OIDC_UI_URL must point to the running OIDC stack");

export default defineConfig({
  testDir: ".",
  testMatch: "oidc/**/*.spec.ts",
  retries: 0,
  workers: 1,
  outputDir: path.join(process.env.E2E_UI_ARTIFACT_DIR || ".", "oidc", "test-results"),
  globalSetup: require.resolve("./oidcSetup"),
  use: {
    ...devices["Desktop Chrome"],
    baseURL,
    storageState: path.join(process.env.E2E_UI_ARTIFACT_DIR || ".", "oidc.storageState.json"),
    trace: "off",
    screenshot: "off",
    video: "off",
  },
});
