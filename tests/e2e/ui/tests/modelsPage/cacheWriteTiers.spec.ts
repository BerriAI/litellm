import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Role, users } from "../../fixtures/users";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";

/**
 * Anthropic bills prompt-cache writes at two rates, a 5 minute TTL and a 1 hour TTL.
 * The dashboard only ever offered the 5 minute field, so an admin who priced a Claude
 * deployment through the UI had every 1 hour cache write billed at the wrong rate.
 * Both tiers must be settable from Add Model and the model edit form, land in
 * `litellm_params`, and read back in the model detail view.
 */
const FIVE_MINUTE_PER_MILLION = 3.75;
const ONE_HOUR_PER_MILLION = 7.5;
const perToken = (perMillion: number): number => perMillion / 1_000_000;
const ONE_HOUR_LABEL = /^Cache Write Cost \(1 hour, per 1M tokens\)/;

const masterKey = users[Role.ProxyAdmin].password;
const auth = { Authorization: `Bearer ${masterKey}` };

async function deleteModel(page: PlaywrightPage, id: string) {
  await page.request.post("/model/delete", { headers: auth, data: { id } });
}

async function readDeployment(page: PlaywrightPage, modelId: string): Promise<Record<string, any> | undefined> {
  const body = await readBack<{ data: Record<string, any>[] }>(page, `/model/info?litellm_model_id=${modelId}`);
  return body.data[0];
}

async function selectProvider(page: PlaywrightPage, providerName: string) {
  const providerDropdown = page.getByRole("combobox", { name: "Provider", exact: true });
  await providerDropdown.click();
  await providerDropdown.fill(providerName);
  await page.getByRole("option").filter({ hasText: providerName }).first().click();
  await expect(providerDropdown).toHaveValue(providerName);
}

test.describe("Cache write cost tiers", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  let createdModelId: string | null = null;

  test.afterEach(async ({ page }) => {
    if (createdModelId) {
      await deleteModel(page, createdModelId);
      createdModelId = null;
    }
  });

  test("the edit form saves the 1 hour rate next to the 5 minute rate and the detail view shows both", async ({
    page,
  }) => {
    const modelName = `e2e-cache-tiers-edit-${Date.now()}`;
    const created = await page.request.post("/model/new", {
      headers: auth,
      data: {
        model_name: modelName,
        litellm_params: {
          model: "anthropic/claude-sonnet-4-6",
          api_key: "sk-e2e-not-used",
          cache_creation_input_token_cost: perToken(FIVE_MINUTE_PER_MILLION),
        },
        model_info: {},
      },
    });
    expect(created.ok(), `POST /model/new for ${modelName}`).toBe(true);
    createdModelId = (await created.json()).model_info?.id;
    expect(createdModelId, "model id from /model/new").toBeTruthy();

    await page.goto("/ui");
    await page.getByText("Models + Endpoints").click();
    const modelIdCell = page.getByTestId(`model-id-${createdModelId}`);
    await expect(modelIdCell).toBeVisible({ timeout: 15_000 });
    await modelIdCell.click();
    await expect(page.getByText("Back to Models").first()).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "Edit Settings" }).click();
    const oneHour = page.getByLabel(ONE_HOUR_LABEL);
    await oneHour.waitFor({ timeout: 15_000 });
    // The cost map already fills a default 1 hour rate in for a known model, so replace it rather than append.
    await oneHour.click({ clickCount: 3 });
    await page.keyboard.press("Delete");
    await page.keyboard.type(String(ONE_HOUR_PER_MILLION));

    const patchBody = await captureRequestBody(
      page,
      { method: "PATCH", urlIncludes: `/model/${createdModelId}/update` },
      async () => {
        await page.getByRole("button", { name: "Save Changes" }).click();
      },
    );
    expect(patchBody.litellm_params.cache_creation_input_token_cost_above_1hr, "1 hour rate sent per token").toBe(
      perToken(ONE_HOUR_PER_MILLION),
    );
    await expect(page.getByText("Model settings updated successfully")).toBeVisible({ timeout: 10_000 });

    const stored = await readDeployment(page, createdModelId!);
    expect(stored?.litellm_params?.cache_creation_input_token_cost_above_1hr, "1 hour rate persisted").toBe(
      perToken(ONE_HOUR_PER_MILLION),
    );
    expect(stored?.litellm_params?.cache_creation_input_token_cost, "5 minute rate untouched").toBe(
      perToken(FIVE_MINUTE_PER_MILLION),
    );

    await expect(page.getByText(ONE_HOUR_LABEL)).toBeVisible();
    await expect(page.getByText(ONE_HOUR_PER_MILLION.toFixed(4))).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText(FIVE_MINUTE_PER_MILLION.toFixed(4))).toBeVisible();
  });

  test("Add Model sends both cache write tiers per token", async ({ page }) => {
    const publicName = `e2e-cache-tiers-add-${Date.now()}`;
    await navigateToPage(page, Page.Models);
    await page.getByRole("tab", { name: "Add Model" }).click();
    await selectProvider(page, "Anthropic");

    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: "Custom Model Name (Enter below)" }).click();
    await page.keyboard.press("Escape");
    await page.getByPlaceholder("Enter custom model name").fill(publicName);
    await page.locator("#api_key").fill("sk-e2e-not-used");

    await page.getByText("Advanced Settings").click();
    await page.getByRole("switch", { name: "Custom Pricing" }).click();
    await page.getByLabel(/^Input Cost \(per 1M tokens\)/).fill("3");
    await page.getByLabel(/^Output Cost \(per 1M tokens\)/).fill("15");
    await page.getByLabel(/^Cache Write Cost \(per 1M tokens\)/).fill(String(FIVE_MINUTE_PER_MILLION));
    await page.getByLabel(ONE_HOUR_LABEL).fill(String(ONE_HOUR_PER_MILLION));

    const body = await captureRequestBody(page, { method: "POST", urlIncludes: "/model/new" }, async () => {
      await page.getByTestId("add-model-btn").click();
    });
    await expect(page.getByText("created successfully")).toBeVisible({ timeout: 15_000 });
    const listed = await readBack<{ data: Record<string, any>[] }>(page, "/v2/model/info");
    createdModelId = listed.data.find((row) => row.model_name === publicName)?.model_info?.id ?? null;

    expect(body.model_name).toBe(publicName);
    expect(body.litellm_params.cache_creation_input_token_cost, "5 minute rate sent per token").toBe(
      perToken(FIVE_MINUTE_PER_MILLION),
    );
    expect(body.litellm_params.cache_creation_input_token_cost_above_1hr, "1 hour rate sent per token").toBe(
      perToken(ONE_HOUR_PER_MILLION),
    );
    expect(createdModelId, "the model landed in the database").toBeTruthy();
  });
});
