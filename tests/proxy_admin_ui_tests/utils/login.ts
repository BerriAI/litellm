import { Page, expect } from "@playwright/test";

export async function loginToUI(page: Page) {
  // Login first
  await page.goto("http://localhost:4000/ui");
  console.log("Navigated to login page");

  // Wait for login form to be visible
  await page.waitForSelector('input[name="username"]', { timeout: 10000 });
  console.log("Login form is visible");

  await page.fill('input[name="username"]', "admin");
  await page.fill('input[name="password"]', "gm");
  console.log("Filled login credentials");

  const loginButton = page.locator('input[type="submit"]');
  await expect(loginButton).toBeEnabled();
  await loginButton.click();
  console.log("Clicked login button");

  // Wait for navigation to complete
  await page.waitForURL("**/*");
}
