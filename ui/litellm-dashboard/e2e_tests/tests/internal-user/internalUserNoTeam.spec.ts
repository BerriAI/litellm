import { test, expect } from "@playwright/test";
import { dismissFeedbackPopup } from "../../helpers/navigation";

/**
 * Logs in fresh inside the test rather than reusing a stored session because
 * this user (seeded with no team memberships) only exists for this one spec —
 * extending globalSetup + the Role enum + the storage-path map for a single
 * assertion isn't worth the maintenance cost.
 */
test.describe("Internal User with no team memberships", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("Create Key team dropdown is empty when the user belongs to no teams", async ({ page }) => {
    // Log in via the form as the no-team seeded user.
    await page.goto("/ui/login");
    await page.getByPlaceholder("Enter your username").fill("noteam@test.local");
    await page.getByPlaceholder("Enter your password").fill("test");
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await expect(page.getByText("Virtual Keys")).toBeVisible({ timeout: 15_000 });
    await dismissFeedbackPopup(page);

    // Open the Create Key modal.
    await page.getByRole("button", { name: /Create New Key/i }).click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    // Click the team dropdown and confirm no team options render. The
    // dropdown's `notFoundContent` is "No teams found" when the user has no
    // memberships and the search hasn't loaded anything.
    const teamSelect = page.locator(".ant-select", { hasText: "Search or select a team" });
    await teamSelect.click();

    const dropdown = page.locator(".ant-select-dropdown:visible").first();
    await expect(dropdown).toBeVisible({ timeout: 5_000 });

    // Either explicit "No teams found" copy or zero option rows — both are
    // valid renderings of an empty dropdown.
    const optionCount = await dropdown.getByRole("option").count();
    expect(optionCount).toBe(0);
  });
});
