import { Page } from "@playwright/test";

export async function loginToUI(page: Page) {
  // Navigate to login page
  await page.goto("/login");

  // Fill in login form
  await page.fill('input[name="username"]', "admin");
  await page.fill('input[name="password"]', "gm");

  // Submit form
  await page.click('button[type="submit"]');

  // Wait for navigation to complete
  await page.waitForURL("**/*");
}
