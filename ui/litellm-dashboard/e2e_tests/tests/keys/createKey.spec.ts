import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Create Key", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Able to create a key with all team models", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByRole("button", { name: "Next" })).toBeVisible();
    await page.getByRole("button", { name: "+ Create New Key" }).click();
    await page.getByTestId("base-input").click();
    await page.getByTestId("base-input").fill("e2eUITestingCreateKeyAllTeamModels");
    await page.locator(".ant-select-selection-overflow").click();
    await page.getByText("All Team Models").click();
    await page.getByRole("combobox", { name: "* Models info-circle :" }).press("Escape");
    await page.getByRole("button", { name: "Create Key" }).click();
    await page.keyboard.press("Escape");
    await expect(page.getByText("e2eUITestingCreateKeyAllTeamModels")).toBeVisible();
  });
});
