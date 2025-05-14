import { test, expect } from "@playwright/test";
import { loginToUI } from "../utils/login";

test.describe("User Info View", () => {
  test.beforeEach(async ({ page }) => {
    await loginToUI(page);
    // Navigate to users page
    await page.goto("http://localhost:4000/ui?page=users");
  });

  test("should display user info when clicking on user ID", async ({
    page,
  }) => {
    // Wait for users table to load
    await page.waitForSelector("table");

    // Get the first user ID cell
    const firstUserIdCell = page.locator(
      "table tbody tr:first-child td:first-child"
    );
    const userId = await firstUserIdCell.textContent();
    console.log("Found user ID:", userId);

    // Click on the user ID
    await firstUserIdCell.click();

    // Wait for user info view to load
    await page.waitForSelector('h1:has-text("User")');
    console.log("User info view loaded");

    // Check for tabs
    await expect(page.locator('button:has-text("Overview")')).toBeVisible();
    await expect(page.locator('button:has-text("Details")')).toBeVisible();

    // Switch to details tab
    await page.locator('button:has-text("Details")').click();

    // Check details section
    await expect(page.locator("text=User ID")).toBeVisible();
    await expect(page.locator("text=Email")).toBeVisible();

    // Go back to users list
    await page.locator('button:has-text("Back to Users")').click();

    // Verify we're back on the users page
    await expect(page.locator('h1:has-text("Users")')).toBeVisible();
  });

  // test("should handle user deletion", async ({ page }) => {
  //   // Wait for users table to load
  //   await page.waitForSelector("table");

  //   // Get the first user ID cell
  //   const firstUserIdCell = page.locator(
  //     "table tbody tr:first-child td:first-child"
  //   );
  //   const userId = await firstUserIdCell.textContent();

  //   // Click on the user ID
  //   await firstUserIdCell.click();

  //   // Wait for user info view to load
  //   await page.waitForSelector('h1:has-text("User")');

  //   // Click delete button
  //   await page.locator('button:has-text("Delete User")').click();

  //   // Confirm deletion in modal
  //   await page.locator('button:has-text("Delete")').click();

  //   // Verify success message
  //   await expect(page.locator("text=User deleted successfully")).toBeVisible();

  //   // Verify we're back on the users page
  //   await expect(page.locator('h1:has-text("Users")')).toBeVisible();

  //   // Verify user is no longer in the table
  //   if (userId) {
  //     await expect(page.locator(`text=${userId}`)).not.toBeVisible();
  //   }
  // });
});
