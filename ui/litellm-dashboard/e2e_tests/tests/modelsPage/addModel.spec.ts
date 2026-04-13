import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_TEAM_CRUD_ID } from "../../constants";
import { Role, users } from "../../fixtures/users";

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

  test("Edit team model TPM and RPM limits", async ({ page }) => {
    const masterKey = users[Role.ProxyAdmin].password;
    const modelName = `e2e-team-model-${Date.now()}`;
    let createdModelId: string | null = null;

    // Create a team-scoped model via API so the test has something to edit
    const createResponse = await page.request.post("/model/new", {
      headers: { Authorization: `Bearer ${masterKey}` },
      data: {
        model_name: modelName,
        litellm_params: {
          model: "openai/fake-gpt-4",
          api_base: "http://127.0.0.1:8090/v1",
          api_key: "fake-key",
          tpm: 100,
          rpm: 200,
        },
        model_info: {
          team_id: E2E_TEAM_CRUD_ID,
        },
      },
    });
    expect(createResponse.ok()).toBe(true);
    const created = await createResponse.json();
    createdModelId = created?.model_info?.id ?? created?.model_id ?? null;
    expect(createdModelId).toBeTruthy();

    try {

      // Navigate to Models + Endpoints
      await page.goto("/ui");
      await page.getByText("Models + Endpoints").click();

      // Click the new model row to open its detail view. The table renders
      // a clickable outer row plus a nested detail row for the same model,
      // so we target the first match (outer row) explicitly.
      const modelRow = page.locator("tr", { hasText: modelName }).first();
      await expect(modelRow).toBeVisible({ timeout: 10_000 });
      await modelRow.click();

      await expect(page.getByText("Back to Models").first()).toBeVisible({ timeout: 10_000 });

      // Edit Settings → change TPM/RPM → Save
      await page.getByRole("button", { name: "Edit Settings" }).click();

      await page.getByPlaceholder("Enter TPM").fill("999");
      await page.getByPlaceholder("Enter RPM").fill("888");

      await page.getByRole("button", { name: "Save Changes" }).click();

      // Verify the new values render back in view mode
      await expect(page.getByText("999", { exact: true })).toBeVisible({ timeout: 10_000 });
      await expect(page.getByText("888", { exact: true })).toBeVisible({ timeout: 10_000 });
    } finally {
      if (createdModelId) {
        await page.request.post("/model/delete", {
          headers: { Authorization: `Bearer ${masterKey}` },
          data: { id: createdModelId },
        });
      }
    }
  });
});
