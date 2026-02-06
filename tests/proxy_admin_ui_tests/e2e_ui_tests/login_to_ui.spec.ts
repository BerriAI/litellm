/*

Login to Admin UI
Basic UI Test

Click on all the tabs ensure nothing is broken
*/

import { test, expect } from "@playwright/test";

test("admin login test", async ({ page }) => {
  // Go to the specified URL
  await page.goto("http://localhost:4000/ui");
  await page.waitForLoadState("networkidle");

  await page.screenshot({ path: "test-results/login_before.png" });

  // Enter "admin" in the username input field
  await page.fill('input[placeholder="Enter your username"]', "admin");

  // Enter "gm" in the password input field
  await page.fill('input[placeholder="Enter your password"]', "gm");

  page.screenshot({ path: "test-results/login_after_inputs.png" });

  // Optionally, you can add an assertion to verify the login button is enabled
  const loginButton = page.getByRole("button", { name: "Login" });
  await expect(loginButton).toBeEnabled();

  // Optionally, you can click the login button to submit the form
  await loginButton.click();
  const tabs = [
    "Virtual Keys",
    "Playground",
    "Models",
    "Usage",
    "Teams",
    "Internal User",
    "Settings",
    "Experimental",
    "API Reference",
    "AI Hub",
  ];

  for (const tab of tabs) {
    const tabElement = page.locator("span.ant-menu-title-content", {
      hasText: tab,
    });
    await tabElement.click();
  }
});
