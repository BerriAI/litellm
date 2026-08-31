import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_TEAM_NO_ADMIN_ID } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup, clickTeamId } from "../../helpers/navigation";

test.describe("Guardrails", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Create a Presidio guardrail, see it in team settings, and delete it", async ({ page }) => {
    const guardrailName = `e2e-presidio-${Date.now()}`;

    await navigateToPage(page, Page.Guardrails);
    await dismissFeedbackPopup(page);

    await page.getByRole("button", { name: /Add New Guardrail/i }).click();
    await page.getByRole("menuitem", { name: "Add Provider Guardrail" }).click();

    const dialog = page.getByRole("dialog", { name: "Create guardrail" });
    await expect(dialog).toBeVisible({ timeout: 10_000 });

    await dialog.getByLabel("Guardrail Name").fill(guardrailName);

    const providerSelect = dialog.getByRole("combobox", { name: "Guardrail Provider" });
    await providerSelect.click();
    await providerSelect.fill("Presidio");
    await page.getByRole("option", { name: "Presidio PII" }).click();

    await dialog.getByLabel("Mode", { exact: true }).click();
    await page.keyboard.type("pre_call");
    await expect(page.getByRole("option", { name: "pre_call" })).toBeAttached({ timeout: 5_000 });
    await page.keyboard.press("Enter");
    await expect(dialog.locator('[data-slot="combobox-chip"]').filter({ hasText: "pre_call" })).toBeVisible({
      timeout: 5_000,
    });
    await dialog.getByText("Create guardrail", { exact: true }).click();

    await dialog.getByLabel("presidio_analyzer_api_base").fill("http://127.0.0.1:9999");
    await expect(dialog.getByLabel("presidio_analyzer_api_base")).toHaveValue("http://127.0.0.1:9999");
    await dialog.getByLabel("presidio_anonymizer_api_base").fill("http://127.0.0.1:9999");
    await expect(dialog.getByLabel("presidio_anonymizer_api_base")).toHaveValue("http://127.0.0.1:9999");

    await dialog.getByRole("button", { name: "Next" }).click();
    await expect(dialog.getByText("Configure PII Protection")).toBeVisible({ timeout: 10_000 });
    await dialog.getByRole("button", { name: "Select All & Mask" }).click();

    await dialog.getByRole("button", { name: "Create Guardrail" }).click();
    await expect(page.getByText("Guardrail created successfully").first()).toBeVisible({ timeout: 15_000 });

    const row = page.locator("table tbody tr").filter({ hasText: guardrailName });
    await expect(row).toHaveCount(1, { timeout: 15_000 });

    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);
    await clickTeamId(page, E2E_TEAM_NO_ADMIN_ID);
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();

    const guardrailsSelect = page.getByRole("combobox", { name: "Select guardrails" });
    await expect(guardrailsSelect).toBeVisible({ timeout: 10_000 });
    await guardrailsSelect.click();
    await guardrailsSelect.fill(guardrailName);
    await expect(page.getByRole("option", { name: guardrailName })).toBeVisible({ timeout: 10_000 });
    await page.keyboard.press("Escape");

    await navigateToPage(page, Page.Guardrails);
    await expect(row).toHaveCount(1, { timeout: 15_000 });
    await row.getByRole("button", { name: "Open guardrail actions" }).click();
    await page.getByRole("menuitem", { name: "Delete" }).click();

    const deleteModal = page.getByRole("dialog", { name: "Delete Guardrail" });
    await expect(deleteModal).toBeVisible({ timeout: 5_000 });
    await deleteModal.getByRole("button", { name: "Delete", exact: true }).click();

    await expect(page.getByText(`Guardrail "${guardrailName}" deleted successfully`)).toBeVisible({
      timeout: 10_000,
    });
    await expect(row).toHaveCount(0, { timeout: 15_000 });

    await page.reload();
    await expect(page.getByRole("button", { name: /Add New Guardrail/i })).toBeVisible({ timeout: 20_000 });
    await expect(page.locator("table tbody tr").filter({ hasText: guardrailName })).toHaveCount(0);
  });
});
