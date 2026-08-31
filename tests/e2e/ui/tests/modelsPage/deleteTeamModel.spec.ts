import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_TEAM_CRUD_ID } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { masterKey } from "../../helpers/traffic";

async function findDeploymentByName(page: PlaywrightPage, modelName: string): Promise<Record<string, any> | undefined> {
  const body = await readBack<{ data: Record<string, any>[] }>(page, "/v2/model/info");
  return body.data.find((row) => row.model_name === modelName);
}

test.describe("Delete team model", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Delete a team-scoped model and verify it leaves the team's model list", async ({ page }) => {
    const modelName = `e2e-team-model-delete-${Date.now()}`;
    const createResponse = await page.request.post("/model/new", {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: {
        model_name: modelName,
        litellm_params: {
          model: "openai/fake-gpt-4",
          api_base: `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`,
          api_key: "fake-key",
        },
        model_info: { team_id: E2E_TEAM_CRUD_ID },
      },
    });
    expect(createResponse.ok(), `/model/new failed: ${createResponse.status()} ${await createResponse.text()}`).toBe(
      true,
    );

    await expect
      .poll(async () => (await findDeploymentByName(page, modelName)) !== undefined, {
        message: `deployment ${modelName} never appeared in /v2/model/info after create`,
        timeout: 30_000,
      })
      .toBe(true);

    await navigateToPage(page, Page.Models);
    await page.getByPlaceholder("Search model names").fill(modelName);

    const row = page.locator("table tbody tr").filter({ hasText: modelName });
    await expect(row).toHaveCount(1, { timeout: 15_000 });
    await expect(row.getByText(E2E_TEAM_CRUD_ID)).toBeVisible({ timeout: 10_000 });

    await row.getByRole("button", { name: "Delete model" }).click();

    const modal = page.getByRole("dialog", { name: "Delete Model" });
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await expect(modal.getByText(modelName).first()).toBeVisible();
    await modal.getByRole("button", { name: "Delete", exact: true }).click();

    await expect(page.getByText("Model deleted successfully").first()).toBeVisible({ timeout: 10_000 });
    await expect(row).toHaveCount(0, { timeout: 15_000 });

    await expect
      .poll(async () => await findDeploymentByName(page, modelName), {
        message: `deployment ${modelName} still readable from /v2/model/info after delete`,
        timeout: 15_000,
      })
      .toBeUndefined();

    await page.reload();
    await page.getByPlaceholder("Search model names").fill(modelName);
    await expect(page.getByText("No models found").first()).toBeVisible({ timeout: 15_000 });
    await expect(page.locator("table tbody tr").filter({ hasText: modelName })).toHaveCount(0);
  });
});
