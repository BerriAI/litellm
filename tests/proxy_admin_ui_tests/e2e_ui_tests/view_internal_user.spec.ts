/*
Test view internal user page
*/

import { test, expect } from "@playwright/test";

test("view internal user page", async ({ page }) => {
  // Go to the specified URL
  await page.goto("http://localhost:4000/ui");

  // Enter "admin" in the username input field
  await page.fill('input[name="username"]', "admin");

  // Enter "gm" in the password input field
  await page.fill('input[name="password"]', "gm");

  // Click the login button
  const loginButton = page.locator('input[type="submit"]');
  await expect(loginButton).toBeEnabled();
  await loginButton.click();

  // Wait for the Internal User tab and click it
  const tabElement = page.locator("span.ant-menu-title-content", {
    hasText: "Internal User",
  });
  await tabElement.click();

  // Wait for the table to load
  await page.waitForSelector("tbody tr", { timeout: 10000 });
  await page.waitForTimeout(2000); // Additional wait for table to stabilize

  // Test all expected fields are present
  // number of keys owned by user
  const keysBadges = page.locator(
    "p.tremor-Badge-text.text-sm.whitespace-nowrap",
    { hasText: "Keys" }
  );
  const keysCountArray = await keysBadges.evaluateAll((elements) =>
    elements.map((el) => {
      const text = el.textContent;
      return text ? parseInt(text.split(" ")[0], 10) : 0;
    })
  );

  const hasNonZeroKeys = keysCountArray.some((count) => count > 0);
  expect(hasNonZeroKeys).toBe(true);

  // test pagination
  // Wait for pagination controls to be visible
  await page.waitForSelector(".flex.justify-between.items-center", {
    timeout: 5000,
  });

  // Check if we're on the first page by looking at the results count
  const resultsText =
    (await page.locator(".text-sm.text-gray-700").textContent()) || "";
  const isFirstPage = resultsText.includes("1 -");

  if (isFirstPage) {
    // On first page, previous button should be disabled
    const prevButton = page.locator("button", { hasText: "Previous" });
    await expect(prevButton).toBeDisabled();
  }

  // Next button should be enabled if there are more pages
  const nextButton = page.locator("button", { hasText: "Next" });
  const totalResults =
    (await page.locator(".text-sm.text-gray-700").textContent()) || "";
  const hasMorePages =
    totalResults.includes("of") && !totalResults.includes("1 - 25 of 25");

  if (hasMorePages) {
    await expect(nextButton).toBeEnabled();
  }
});
