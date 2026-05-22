import { chromium, expect } from "@playwright/test";
import { users, Role, STORAGE_PATHS } from "./fixtures/users";
import * as fs from "fs";

async function globalSetup() {
  const browser = await chromium.launch();

  for (const role of Object.values(Role)) {
    const { email, password } = users[role];
    const storagePath = STORAGE_PATHS[role];
    const page = await browser.newPage();
    try {
      await page.goto("http://localhost:4000/ui/login");
      await page.getByPlaceholder("Enter your username").fill(email);
      await page.getByPlaceholder("Enter your password").fill(password);
      await page.getByRole("button", { name: "Login", exact: true }).click();
      await page.waitForURL(
        (url) => url.pathname.startsWith("/ui") && !url.pathname.includes("/login"),
        { timeout: 30_000 },
      );
      await expect(page.locator("a", { hasText: "Virtual Keys" })).toBeVisible({ timeout: 30_000 });
      // Dismiss feedback popup if present
      const dismiss = page.getByText("Don't ask me again");
      if (await dismiss.isVisible({ timeout: 1_500 }).catch(() => false)) {
        await dismiss.click();
      }
      await page.context().storageState({ path: storagePath });
    } catch (e) {
      fs.mkdirSync("test-results", { recursive: true });
      await page.screenshot({ path: `test-results/global-setup-${role}-failure.png`, fullPage: true });
      console.error(`Global setup failed for role ${role}. Screenshot saved. URL: ${page.url()}`);
      throw e;
    } finally {
      await page.close();
    }
  }

  await browser.close();
}

export default globalSetup;
