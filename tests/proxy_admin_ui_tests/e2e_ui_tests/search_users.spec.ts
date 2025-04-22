/*
Search Users in Admin UI
E2E Test for user search functionality

Tests:
1. Navigate to Internal Users tab
2. Verify search input exists
3. Test search functionality
4. Verify results update
*/

import { test, expect } from "@playwright/test";

test("user search test", async ({ page }) => {
  // Set a longer timeout for the entire test
  test.setTimeout(60000);

  // Enable console logging
  page.on("console", (msg) => console.log("PAGE LOG:", msg.text()));

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

  // Wait for navigation to complete and dashboard to load
  await page.waitForLoadState("networkidle");
  console.log("Page loaded after login");

  // Take a screenshot for debugging
  await page.screenshot({ path: "after-login.png" });
  console.log("Took screenshot after login");

  // Try to find the Internal User tab with more debugging
  console.log("Looking for Internal User tab...");
  const internalUserTab = page.locator("span.ant-menu-title-content", {
    hasText: "Internal User",
  });

  // Wait for the tab to be visible
  await internalUserTab.waitFor({ state: "visible", timeout: 10000 });
  console.log("Internal User tab is visible");

  // Take another screenshot before clicking
  await page.screenshot({ path: "before-tab-click.png" });
  console.log("Took screenshot before tab click");

  await internalUserTab.click();
  console.log("Clicked Internal User tab");

  // Wait for the page to load and table to be visible
  await page.waitForSelector("tbody tr", { timeout: 10000 });
  await page.waitForTimeout(2000); // Additional wait for table to stabilize
  console.log("Table is visible");

  // Take a final screenshot
  await page.screenshot({ path: "after-tab-click.png" });
  console.log("Took screenshot after tab click");

  // Verify search input exists
  const searchInput = page.locator('input[placeholder="Search by email..."]');
  await expect(searchInput).toBeVisible();
  console.log("Search input is visible");

  // Test search functionality
  const initialUserCount = await page.locator("tbody tr").count();
  console.log(`Initial user count: ${initialUserCount}`);

  // Perform a search
  const testEmail = "test@";
  await searchInput.fill(testEmail);
  console.log("Filled search input");

  // Wait for the debounced search to complete
  await page.waitForTimeout(500);
  console.log("Waited for debounce");

  // Wait for the results count to update
  await page.waitForFunction((initialCount) => {
    const currentCount = document.querySelectorAll("tbody tr").length;
    return currentCount !== initialCount;
  }, initialUserCount);
  console.log("Results updated");

  const filteredUserCount = await page.locator("tbody tr").count();
  console.log(`Filtered user count: ${filteredUserCount}`);

  expect(filteredUserCount).toBeDefined();

  // Clear the search
  await searchInput.clear();
  console.log("Cleared search");

  await page.waitForTimeout(500);
  console.log("Waited for debounce after clear");

  await page.waitForFunction((initialCount) => {
    const currentCount = document.querySelectorAll("tbody tr").length;
    return currentCount === initialCount;
  }, initialUserCount);
  console.log("Results reset");

  const resetUserCount = await page.locator("tbody tr").count();
  console.log(`Reset user count: ${resetUserCount}`);

  expect(resetUserCount).toBe(initialUserCount);
});
