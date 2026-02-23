import { Page, expect } from "@playwright/test";

export async function loginToUI(page: Page) {
  // Login first
  await page.goto("http://localhost:4000/ui");
  await page.waitForLoadState("networkidle");
  console.log("Navigated to login page");

  page.screenshot({ path: "test-results/login_utils_before.png" });
  // Wait for login form to be visible
  await page.waitForSelector('input[placeholder="Enter your username"]', {
    timeout: 10000,
  });
  console.log("Login form is visible");

  await page.fill('input[placeholder="Enter your username"]', "admin");
  await page.fill('input[placeholder="Enter your password"]', "gm");
  console.log("Filled login credentials");

  const loginButton = page.getByRole("button", { name: "Login" });
  await expect(loginButton).toBeEnabled();
  await loginButton.click();
  console.log("Clicked login button");

  // Wait for navigation to complete
  await page.waitForURL("**/*");
}
