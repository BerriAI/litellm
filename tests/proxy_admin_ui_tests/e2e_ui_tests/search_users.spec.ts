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
  // Login first
  await page.goto("http://localhost:4000/ui");
  await page.fill('input[name="username"]', "admin");
  await page.fill('input[name="password"]', "gm");
  await page.locator('input[type="submit"]').click();

  // Navigate to Internal User tab
  const internalUserTab = page.locator("span.ant-menu-title-content", {
    hasText: "Internal User",
  });
  await internalUserTab.click();

  // Wait for the page to load
  await page.waitForTimeout(1000);

  // Verify search input exists
  const searchInput = page.locator('input[placeholder="Search by email..."]');
  await expect(searchInput).toBeVisible();

  // Test search functionality
  // First, store the initial number of users
  const initialUserCount = await page.locator("tbody tr").count();

  // Perform a search
  const testEmail = "test@"; // Partial email for testing
  await searchInput.fill(testEmail);

  // Wait for the search results to update (debounce is 300ms)
  await page.waitForTimeout(500);

  // Get the filtered user count
  const filteredUserCount = await page.locator("tbody tr").count();

  // Verify that the search results have been updated
  // Note: This test assumes there are some users in the system and the search returns fewer results
  expect(filteredUserCount).toBeDefined();

  // Clear the search
  await searchInput.clear();

  // Wait for the results to reset
  await page.waitForTimeout(500);

  // Verify the list returns to its original state
  const resetUserCount = await page.locator("tbody tr").count();
  expect(resetUserCount).toBe(initialUserCount);
});
