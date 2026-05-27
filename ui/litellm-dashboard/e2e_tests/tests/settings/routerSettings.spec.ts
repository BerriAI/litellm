import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";

test.describe("Router Settings - Fallbacks", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Add a fallback and verify it appears in the table", async ({ page }) => {
    await navigateToPage(page, Page.RouterSettings);

    // Three tabs: Loadbalancing / Routing Groups / Fallbacks / General — click Fallbacks
    await page.getByRole("tab", { name: "Fallbacks" }).click();

    // Open the Add Fallbacks modal
    await page.getByRole("button", { name: /Add Fallbacks/i }).click();

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });

    // Pick the primary model: fake-openai-gpt-4 (seeded in fixtures/config.yml)
    const primarySelect = modal.locator(".ant-select").filter({ hasText: "Select primary model" });
    await primarySelect.click();
    await page.locator(".ant-select-dropdown:visible").getByTitle("fake-openai-gpt-4").click();

    // Pick a fallback: fake-anthropic-claude
    const fallbackSelect = modal.locator(".ant-select").filter({ hasText: "Select fallback models" });
    await fallbackSelect.click();
    await page.locator(".ant-select-dropdown:visible").getByTitle("fake-anthropic-claude").click();
    await page.keyboard.press("Escape");

    // Save
    await modal.getByRole("button", { name: /Save All Configurations/i }).click();

    // Success toast
    await expect(page.getByText(/fallback configuration\(s\) added successfully/i).first())
      .toBeVisible({ timeout: 10_000 });

    // Modal closes, the new row shows up with both models
    await expect(modal).not.toBeVisible({ timeout: 5_000 });

    const tableBody = page.locator("table tbody");
    await expect(tableBody.getByText("fake-openai-gpt-4").first()).toBeVisible({ timeout: 10_000 });
    await expect(tableBody.getByText("fake-anthropic-claude").first()).toBeVisible({ timeout: 10_000 });
  });
});
