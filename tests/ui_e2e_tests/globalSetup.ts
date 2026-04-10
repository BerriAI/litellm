import { chromium, expect } from "@playwright/test";
import { users, Role, ADMIN_STORAGE_PATH } from "./constants";
import * as fs from "fs";

async function globalSetup() {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto("http://localhost:4000/ui/login");
  await page.getByPlaceholder("Enter your username").fill(users[Role.ProxyAdmin].email);
  await page.getByPlaceholder("Enter your password").fill(users[Role.ProxyAdmin].password);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  try {
    // Wait for navigation away from login page into the dashboard
    await page.waitForURL(
      (url) => url.pathname.startsWith("/ui") && !url.pathname.includes("/login"),
      { timeout: 30_000 },
    );
    // Wait for sidebar to render as a signal that the dashboard is ready
    await expect(page.getByRole("menuitem", { name: "Virtual Keys" })).toBeVisible({ timeout: 30_000 });
  } catch (e) {
    // Save a screenshot for debugging before re-throwing
    fs.mkdirSync("test-results", { recursive: true });
    await page.screenshot({ path: "test-results/global-setup-failure.png", fullPage: true });
    console.error("Global setup failed. Screenshot saved to test-results/global-setup-failure.png");
    console.error("Current URL:", page.url());
    throw e;
  }
  await page.context().storageState({ path: ADMIN_STORAGE_PATH });
  await browser.close();
}

export default globalSetup;
