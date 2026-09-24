import { test, expect, Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

async function goToInternalUsers(page: PlaywrightPage) {
  await navigateToPage(page, Page.Users);
  await expect(page.getByRole("columnheader", { name: "User ID" })).toBeVisible({ timeout: 30_000 });
  await expect(userRows(page)).not.toHaveCount(0, { timeout: 30_000 });
}

const userRows = (page: PlaywrightPage) => page.getByRole("row").filter({ has: page.getByRole("cell") });

test.describe("Internal Users Page", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("lists the seeded users under the identifying columns", async ({ page }) => {
    await goToInternalUsers(page);

    await expect(page.getByRole("columnheader", { name: "User ID" })).toBeVisible();
    await expect(page.getByRole("columnheader", { name: "Virtual Keys" })).toBeVisible();
  });

  test("cannot page backwards off the first page", async ({ page }) => {
    await goToInternalUsers(page);

    await expect(page.getByRole("button", { name: "Go to previous page" })).toBeDisabled();
  });
});
