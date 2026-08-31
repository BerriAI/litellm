import { test, expect, Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

const userRows = (page: PlaywrightPage) => page.getByRole("row").filter({ has: page.getByRole("cell") });

async function goToInternalUsers(page: PlaywrightPage) {
  await navigateToPage(page, Page.Users);
  await expect(page.getByRole("columnheader", { name: "User ID" })).toBeVisible({ timeout: 30_000 });
  await expect(userRows(page)).not.toHaveCount(0, { timeout: 30_000 });
}

test.describe("Internal Users Search", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("narrows the table to the matching email, and restores it when cleared", async ({ page }) => {
    await goToInternalUsers(page);

    const search = page.getByPlaceholder("Search by email…");
    await expect(search).toBeVisible();

    await search.fill("noteam@");
    await expect(userRows(page)).toHaveCount(1, { timeout: 30_000 });
    await expect(userRows(page).first()).toContainText("noteam@test.local");

    await search.clear();
    await expect(userRows(page).filter({ hasText: "admin@test.local" })).not.toHaveCount(0, { timeout: 30_000 });
  });

  test("filters the table down to one user by user ID", async ({ page }) => {
    await goToInternalUsers(page);

    await page.getByRole("button", { name: "Filters" }).click();
    await page.getByTestId("users-filter-user-id").fill("e2e-internal-noteam");
    await page.getByTestId("filter-drawer-apply").click();

    await expect(userRows(page)).toHaveCount(1, { timeout: 30_000 });
    await expect(userRows(page).first()).toContainText("noteam@test.local");
  });

  test("shows no users when the SSO ID matches nobody", async ({ page }) => {
    await goToInternalUsers(page);

    await page.getByRole("button", { name: "Filters" }).click();
    await page.getByTestId("users-filter-sso-id").fill("e2e-sso-id-that-matches-nobody");
    await page.getByTestId("filter-drawer-apply").click();

    await expect(userRows(page)).toHaveCount(0, { timeout: 30_000 });
  });
});
