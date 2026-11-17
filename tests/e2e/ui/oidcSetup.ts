import { chromium, expect } from "@playwright/test";
import * as fs from "fs";
import * as path from "path";

export default async function oidcSetup() {
  const baseURL = process.env.E2E_OIDC_UI_URL;
  const issuer = process.env.JWT_ISSUER;
  const username = process.env.E2E_OIDC_USERNAME;
  const password = process.env.E2E_OIDC_PASSWORD;
  if (!baseURL || !issuer || !username || !password) {
    throw new Error("The OIDC setup requires a running stack, issuer, and provisioned actor credentials");
  }
  const artifactDir = process.env.E2E_UI_ARTIFACT_DIR || ".";
  fs.mkdirSync(artifactDir, { recursive: true });
  const browser = await chromium.launch();
  try {
    const page = await browser.newPage();
    await page.goto(`${baseURL.replace(/\/$/, "")}/sso/key/generate`);
    await expect(page).toHaveURL(new RegExp(`^${issuer.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}/`));
    await page.getByLabel("Username or email").fill(username);
    await page.getByLabel("Password", { exact: true }).fill(password);
    await page.getByRole("button", { name: "Sign In", exact: true }).click();
    await page.waitForURL((url) => url.origin === new URL(baseURL).origin && url.pathname.startsWith("/ui"));
    const statePath = path.join(artifactDir, "oidc.storageState.json");
    await page.context().storageState({ path: statePath });
    fs.chmodSync(statePath, 0o600);
  } finally {
    await browser.close();
  }
}
