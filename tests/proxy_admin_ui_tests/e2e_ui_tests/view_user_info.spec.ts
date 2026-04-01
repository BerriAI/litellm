import { test, expect } from "@playwright/test";
import { loginToUI } from "../utils/login";

test.describe("User Info View", () => {
  test("should display user info when clicking on user ID", async ({
    page,
  }) => {
    await page.goto("http://localhost:4000/ui");
    await page.waitForLoadState("networkidle");

    page.screenshot({
      path: "test-results/view_user_info_before_login.png",
    });

    // Enter "admin" in the username input field
    await page.fill('input[placeholder="Enter your username"]', "admin");
    page.screenshot({
      path: "test-results/view_user_info_after_username_input.png",
    });

    // Enter "gm" in the password input field
    await page.fill('input[placeholder="Enter your password"]', "gm");
    page.screenshot({
      path: "test-results/view_user_info_after_password_input.png",
    });

    // Click the login button
    const loginButton = page.getByRole("button", { name: "Login" });
    await expect(loginButton).toBeEnabled();
    await loginButton.click();
    page.screenshot({
      path: "test-results/view_user_info_after_login_button_click.png",
    });

    // Wait for navigation to complete and dashboard to load
    await page.waitForLoadState("networkidle");
    const tabElement = page.locator("span.ant-menu-title-content", {
      hasText: "Internal User",
    });
    await tabElement.click();
    page.screenshot({
      path: "test-results/view_user_info_after_internal_user_tab_click.png",
    });
    // Wait for loading state to disappear
    await page.waitForSelector('text="🚅 Loading users..."', {
      state: "hidden",
      timeout: 10000,
    });
    page.screenshot({ path: "test-results/view_user_info_after_loading.png" });
    // Wait for users table to load
    await page.waitForSelector("table");
    page.screenshot({
      path: "test-results/view_user_info_after_table_load.png",
    });
    // Get the first user ID cell
    const firstUserIdCell = page.locator(
      "table tbody tr:first-child td:first-child"
    );
    const userId = await firstUserIdCell.textContent();
    console.log("Found user ID:", userId);

    // Click on the user ID
    await firstUserIdCell.click();
    await page.waitForLoadState("networkidle");

    // Check for tabs
    await expect(page.locator('button:has-text("Overview")')).toBeVisible({
      timeout: 10000,
    });
    await expect(page.locator('button:has-text("Details")')).toBeVisible({
      timeout: 10000,
    });

    // Switch to details tab
    await page.locator('button:has-text("Details")').click();

    // Check details section
    await expect(page.locator("text=User ID")).toBeVisible();
    await expect(page.locator("text=Email")).toBeVisible();

    // Go back to users list
    await page.locator('button:has-text("Back to Users")').click();

    // Verify we're back on the users page
    await expect(page.locator("table")).toBeVisible();
    await expect(
      page.locator('input[placeholder="Search by email..."]')
    ).toBeVisible();
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
