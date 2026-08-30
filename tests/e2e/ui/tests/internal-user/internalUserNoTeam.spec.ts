import { test, expect } from "@playwright/test";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";

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
    await expect(page.getByRole("complementary").getByText("Virtual Keys")).toBeVisible({ timeout: 30_000 });
    expect(new URL(page.url()).pathname).not.toMatch(/\/connect$/);
    await navigateToPage(page, Page.ApiKeys);
    // Scope to the sidebar; the top-bar breadcrumb also shows "Virtual Keys".
    await expect(page.getByRole("complementary").getByText("Virtual Keys")).toBeVisible({ timeout: 15_000 });

    // Open the Create Key modal.
    await page.getByRole("button", { name: /Create New Key/i }).click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    const teamSelect = page.getByTestId("team-dropdown").getByRole("combobox");
    await teamSelect.click();

    const dropdown = page.locator('[data-slot="combobox-content"]:visible').first();
    await expect(dropdown).toBeVisible({ timeout: 5_000 });

    // Wait for the settled-empty state, not a transient one. The dropdown shows
    // "Loading teams…" while teams load and only swaps in "No teams found" once
    // the request resolves with nothing (team_dropdown.tsx passes both copies to
    // PaginatedSearchSelect). Asserting on it means a regression where teams DO
    // load for this user fails here instead of racing a one-shot count() against
    // an in-flight request.
    await expect(dropdown.getByText("No teams found")).toBeVisible({ timeout: 10_000 });
    await expect(dropdown.getByRole("option")).toHaveCount(0);
  });
});
