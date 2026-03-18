import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_REGENERATE_KEY_ID_PREFIX } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Regenerate Key", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Able to regenerate a key", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
    await page
      .locator("button", {
        hasText: E2E_REGENERATE_KEY_ID_PREFIX,
      })
      .click();
    await page.getByRole("button", { name: "Regenerate Key" }).click();
    await page.getByRole("button", { name: "Regenerate", exact: true }).click();
    await expect(page.getByText("Virtual Key regenerated")).toBeVisible();
  });
});
