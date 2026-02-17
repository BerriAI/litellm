import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_UPDATE_LIMITS_KEY_ID_PREFIX } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Update Key TPM and RPM Limits", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Able to update a key's TPM and RPM limits", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
    await page
      .locator("button", {
        hasText: E2E_UPDATE_LIMITS_KEY_ID_PREFIX,
      })
      .click();
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();
    await page.getByRole("spinbutton", { name: "TPM Limit" }).click();
    await page.getByRole("spinbutton", { name: "TPM Limit" }).fill("123");
    await page.getByRole("spinbutton", { name: "RPM Limit" }).click();
    await page.getByRole("spinbutton", { name: "RPM Limit" }).fill("456");
    await page.getByRole("button", { name: "Save Changes" }).click();
    await expect(page.getByRole("paragraph").filter({ hasText: "TPM: 123" })).toBeVisible();
    await expect(page.getByRole("paragraph").filter({ hasText: "RPM: 456" })).toBeVisible();
  });
});
