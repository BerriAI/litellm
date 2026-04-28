import { test, expect } from "@playwright/test";
import {
  ADMIN_VIEWER_STORAGE_PATH,
  E2E_INTERNAL_USER_KEY_ALIAS,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Proxy Admin Viewer - Keys (read-only)", () => {
  test.use({ storageState: ADMIN_VIEWER_STORAGE_PATH });

  test("Sees the keys table without an Access Denied gate", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    // Hard entry-level gate is gone.
    await expect(page.getByText("Access Denied")).toHaveCount(0);
    await expect(
      page.getByText("Ask your proxy admin for access to create keys"),
    ).toHaveCount(0);

    // Admin viewer has the same all-keys view as proxy admin
    // (backend use_substring_matching path).
    await expect(page.getByText(E2E_INTERNAL_USER_KEY_ALIAS)).toBeVisible({ timeout: 10_000 });

    // rolesWithWriteAccess guard hides the create affordance.
    await expect(page.getByRole("button", { name: /Create New Key/i })).toHaveCount(0);
  });

  test("Can open a key detail view but cannot regenerate / delete / edit", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    // Open the detail view of one of the seeded keys.
    const keyRow = page.locator("tr", { hasText: E2E_INTERNAL_USER_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();

    // Detail view loaded.
    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    // No write affordances on the detail view.
    await expect(page.getByRole("button", { name: "Regenerate Key" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Delete Key" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Edit Settings" })).toHaveCount(0);
  });
});
