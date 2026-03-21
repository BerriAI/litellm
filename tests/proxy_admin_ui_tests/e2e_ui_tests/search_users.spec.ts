/*
Search Users in Admin UI
E2E Test for user search functionality

Tests:
1. Navigate to Internal Users tab
2. Verify search input exists
3. Test search functionality
4. Verify results update
5. Test filtering by email, user ID, and SSO user ID
*/

import { test, expect } from "@playwright/test";

test("user search test", async ({ page }) => {
  // Set a longer timeout for the entire test
  test.setTimeout(60000);

  // Enable console logging
  page.on("console", (msg) => console.log("PAGE LOG:", msg.text()));

  // Login first
  await page.goto("http://localhost:4000/ui");
  await page.waitForLoadState("networkidle");
  console.log("Navigated to login page");

  page.screenshot({ path: "test-results/search_users_before_login.png" });

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
  await page.waitForSelector("tbody tr", { timeout: 30000 });
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

test("user filter test", async ({ page }) => {
  // Set a longer timeout for the entire test
  test.setTimeout(60000);

  // Enable console logging
  page.on("console", (msg) => console.log("PAGE LOG:", msg.text()));

  // Login first
  await page.goto("http://localhost:4000/ui");
  await page.waitForLoadState("networkidle");
  console.log("Navigated to login page");

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

  // Wait for navigation to complete and dashboard to load
  await page.waitForLoadState("networkidle");
  console.log("Page loaded after login");

  // Navigate to Internal Users tab
  const internalUserTab = page.locator("span.ant-menu-title-content", {
    hasText: "Internal User",
  });
  await internalUserTab.waitFor({ state: "visible", timeout: 10000 });
  await internalUserTab.click();
  console.log("Clicked Internal User tab");

  // Wait for the page to load and table to be visible
  await page.waitForSelector("tbody tr", { timeout: 30000 });
  await page.waitForTimeout(2000); // Additional wait for table to stabilize
  console.log("Table is visible");

  // Get initial user count
  const initialUserCount = await page.locator("tbody tr").count();
  console.log(`Initial user count: ${initialUserCount}`);

  // Click the filter button to show additional filters
  const filterButton = page.getByRole("button", {
    name: "Filters",
    exact: true,
  });
  await filterButton.click();
  console.log("Clicked filter button");
  await page.waitForTimeout(500); // Wait for filters to appear

  // Test user ID filter
  const userIdInput = page.locator('input[placeholder="Filter by User ID"]');
  await expect(userIdInput).toBeVisible();
  console.log("User ID filter is visible");

  await userIdInput.fill("user");
  console.log("Filled user ID filter");
  await page.waitForTimeout(1000);
  const userIdFilteredCount = await page.locator("tbody tr").count();
  console.log(`User ID filtered count: ${userIdFilteredCount}`);
  expect(userIdFilteredCount).toBeLessThan(initialUserCount);

  // Clear user ID filter
  await userIdInput.clear();
  await page.waitForTimeout(1000);
  console.log("Cleared user ID filter");

  // Test SSO user ID filter
  const ssoUserIdInput = page.locator('input[placeholder="Filter by SSO ID"]');
  await expect(ssoUserIdInput).toBeVisible();
  console.log("SSO user ID filter is visible");

  await ssoUserIdInput.fill("sso");
  console.log("Filled SSO user ID filter");
  await page.waitForTimeout(1000);
  const ssoUserIdFilteredCount = await page.locator("tbody tr").count();
  console.log(`SSO user ID filtered count: ${ssoUserIdFilteredCount}`);
  expect(ssoUserIdFilteredCount).toBeLessThan(initialUserCount);

  // Clear SSO user ID filter
  await ssoUserIdInput.clear();
  await page.waitForTimeout(5000);
  console.log("Cleared SSO user ID filter");

  // Verify count returns to initial after clearing all filters
  const finalUserCount = await page.locator("tbody tr").count();
  console.log(`Final user count: ${finalUserCount}`);
  expect(finalUserCount).toBe(initialUserCount);
});
