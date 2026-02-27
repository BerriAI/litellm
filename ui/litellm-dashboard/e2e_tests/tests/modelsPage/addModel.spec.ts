import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

test.describe("Add Model", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Able to see all models for a specific provider in the model dropdown", async ({ page }) => {
    await page.goto("/ui");

    await page.getByText("Models + Endpoints").click();
    await page.getByRole("tab", { name: "Add Model" }).click();

    const providerInputDropdown = page.getByRole("combobox", { name: /Provider/i });
    await providerInputDropdown.fill("Anthropic");
    await page.waitForTimeout(1000);
    await providerInputDropdown.press("Enter");
    await page.waitForTimeout(2000);

    const providerModelsDropdown = page.locator(".ant-select-selection-overflow").first();
    await providerModelsDropdown.click();
    await expect(page.getByTitle("claude-haiku-4-5", { exact: true })).toBeVisible();
  });
});
