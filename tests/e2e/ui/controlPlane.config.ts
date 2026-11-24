import { defineConfig, devices } from "@playwright/test";
import path from "node:path";
import { ARTIFACT_DIR } from "./constants";

const baseURL = (process.env.CONTROL_PLANE_E2E_URL ?? "http://127.0.0.1:4100").replace(/\/?$/, "/");

export default defineConfig({
  testDir: "./controlPlane",
  testMatch: "*.spec.ts",
  workers: 1,
  timeout: 90_000,
  reporter: "line",
  outputDir: path.join(ARTIFACT_DIR, "control-plane"),
  globalSetup: require.resolve("./controlPlane/setup"),
  use: {
    ...devices["Desktop Chrome"],
    baseURL,
  },
});
