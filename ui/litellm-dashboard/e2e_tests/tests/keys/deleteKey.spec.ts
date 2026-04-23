import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_DELETE_KEY_ID_PREFIX, E2E_DELETE_KEY_NAME } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Delete Key", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Able to delete a key", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
    await page
      .locator("button", {
        hasText: E2E_DELETE_KEY_ID_PREFIX,
      })
      .click();
    await page.getByRole("button", { name: "Delete Key" }).click();
    await page.getByRole("textbox", { name: E2E_DELETE_KEY_NAME }).click();
    await page.getByRole("textbox", { name: E2E_DELETE_KEY_NAME }).fill(E2E_DELETE_KEY_NAME);
    const deleteButton = page.getByRole("button", { name: "Delete", exact: true });
    await expect(deleteButton).toBeEnabled();
    await deleteButton.click();
    await expect(page.getByText("Key deleted successfully")).toBeVisible();
  });
});
