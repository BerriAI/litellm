import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Role, users } from "../../fixtures/users";

/**
 * Regression: clearing the Input / Output / Cache Read / Cache Write Cost
 * fields on a deployment with a user-set pricing override must actually remove
 * the override from both `litellm_params` and `model_info`.
 *
 * Pre-fix, the UI sent the old pricing back on every save (the spread of
 * `values.litellm_params` re-injected it), and the backend's `exclude_none=True`
 * stripped any null that did make it through. End-result: the dashboard
 * displayed "Saved" but the override remained in the DB. The cache fields had
 * the same bug in a parallel code path and are covered here too.
 */
test.describe("Clear custom pricing on a deployment", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  const masterKey = users[Role.ProxyAdmin].password;
  const SEED_INPUT_PER_TOKEN = 0.0000777;
  const SEED_OUTPUT_PER_TOKEN = 0.0000999;
  const SEED_CACHE_READ_PER_TOKEN = 0.0000333;
  const SEED_CACHE_WRITE_PER_TOKEN = 0.0000555;

  // Unique-per-run name so concurrent / repeated runs don't collide on the
  // shared dashboard DB. Captured here so afterEach can clean it up.
  let createdModelId: string | null = null;
  let modelName: string;

  test.beforeEach(async ({ page }) => {
    modelName = `e2e-clear-pricing-${Date.now()}`;
    const res = await page.request.post("/model/new", {
      headers: { Authorization: `Bearer ${masterKey}` },
      data: {
        model_name: modelName,
        litellm_params: {
          model: "openai/gpt-4o",
          api_key: "sk-e2e-not-used",
          input_cost_per_token: SEED_INPUT_PER_TOKEN,
          output_cost_per_token: SEED_OUTPUT_PER_TOKEN,
          cache_read_input_token_cost: SEED_CACHE_READ_PER_TOKEN,
          cache_creation_input_token_cost: SEED_CACHE_WRITE_PER_TOKEN,
        },
        model_info: {},
      },
    });
    expect(res.ok(), `POST /model/new for ${modelName}`).toBe(true);
    const body = await res.json();
    createdModelId = body.model_info?.id ?? body.model_id;
    expect(createdModelId, "model id from /model/new").toBeTruthy();
  });

  test.afterEach(async ({ page }) => {
    // The dashboard DB persists across this suite (not just per-test), so every
    // model created here must be cleaned up regardless of test outcome.
    if (createdModelId) {
      await page.request.post("/model/delete", {
        headers: { Authorization: `Bearer ${masterKey}` },
        data: { id: createdModelId },
      });
      createdModelId = null;
    }
  });

  test("UI sends null for cleared pricing and backend removes the override", async ({
    page,
  }) => {
    // Navigate to the model detail view.
    await page.goto("/ui");
    await page.getByText("Models + Endpoints").click();

    const modelRow = page.locator("tr", { hasText: modelName }).first();
    await expect(modelRow).toBeVisible({ timeout: 15_000 });
    await modelRow.click();
    await expect(page.getByText("Back to Models").first()).toBeVisible({
      timeout: 10_000,
    });

    // Sanity: the seeded pricing is shown in the detail view (77.7000 / 99.9000
    // per 1M tokens). The dashboard renders the per-token rate × 1e6.
    await expect(page.getByText("77.7000")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("99.9000")).toBeVisible({ timeout: 10_000 });

    // Open the edit form and clear all four pricing fields.
    await page.getByRole("button", { name: "Edit Settings" }).click();
    const inputCost = page.getByPlaceholder("Enter input cost");
    const outputCost = page.getByPlaceholder("Enter output cost");
    // Both cache fields share the same placeholder ("Defaults to Input Cost if blank"),
    // so disambiguate via the Form.Item id (AntD assigns the `name` prop as input id).
    const cacheReadCost = page.locator("#cache_read_cost");
    const cacheWriteCost = page.locator("#cache_write_cost");
    await inputCost.waitFor({ timeout: 15_000 });
    for (const field of [inputCost, outputCost, cacheReadCost, cacheWriteCost]) {
      await field.click({ clickCount: 3 });
      await page.keyboard.press("Delete");
    }

    // Capture the outgoing PATCH so we can assert the UI sends explicit nulls.
    const patchPromise = page.waitForRequest(
      (req) =>
        req.method() === "PATCH" &&
        req.url().includes(`/model/${createdModelId}/update`)
    );
    await page.getByRole("button", { name: "Save Changes" }).click();
    const patchReq = await patchPromise;
    const patchBody = JSON.parse(patchReq.postData() ?? "{}");
    expect(
      patchBody.litellm_params.input_cost_per_token,
      "UI sends explicit null for cleared input cost"
    ).toBeNull();
    expect(
      patchBody.litellm_params.output_cost_per_token,
      "UI sends explicit null for cleared output cost"
    ).toBeNull();
    expect(
      patchBody.litellm_params.cache_read_input_token_cost,
      "UI sends explicit null for cleared cache_read cost"
    ).toBeNull();
    expect(
      patchBody.litellm_params.cache_creation_input_token_cost,
      "UI sends explicit null for cleared cache_write cost"
    ).toBeNull();

    // Success toast confirms the save was accepted.
    await expect(
      page.getByText("Model settings updated successfully")
    ).toBeVisible({ timeout: 10_000 });

    // Verify via the management API: the user-set rate is gone from both blobs.
    // The cost-map may synthesize a default for known providers in the response,
    // so the assertion is "no longer the seeded value" rather than literally
    // undefined.
    const infoRes = await page.request.get(
      `/v2/model/info?include_team_models=true&page=1&size=100&modelId=${createdModelId}`,
      { headers: { Authorization: `Bearer ${masterKey}` } }
    );
    expect(infoRes.ok()).toBe(true);
    const infoBody = await infoRes.json();
    const row = (infoBody.data ?? infoBody).find?.(
      (m: any) => m?.model_info?.id === createdModelId
    );
    expect(row, "model info row").toBeTruthy();

    expect(
      "input_cost_per_token" in row.litellm_params,
      "litellm_params.input_cost_per_token key removed"
    ).toBe(false);
    expect(
      "output_cost_per_token" in row.litellm_params,
      "litellm_params.output_cost_per_token key removed"
    ).toBe(false);
    expect(
      "cache_read_input_token_cost" in row.litellm_params,
      "litellm_params.cache_read_input_token_cost key removed"
    ).toBe(false);
    expect(
      "cache_creation_input_token_cost" in row.litellm_params,
      "litellm_params.cache_creation_input_token_cost key removed"
    ).toBe(false);
    expect(
      row.model_info.input_cost_per_token,
      "model_info.input_cost_per_token no longer the seeded override"
    ).not.toBe(SEED_INPUT_PER_TOKEN);
    expect(
      row.model_info.output_cost_per_token,
      "model_info.output_cost_per_token no longer the seeded override"
    ).not.toBe(SEED_OUTPUT_PER_TOKEN);
    expect(
      row.model_info.cache_read_input_token_cost,
      "model_info.cache_read_input_token_cost no longer the seeded override"
    ).not.toBe(SEED_CACHE_READ_PER_TOKEN);
    expect(
      row.model_info.cache_creation_input_token_cost,
      "model_info.cache_creation_input_token_cost no longer the seeded override"
    ).not.toBe(SEED_CACHE_WRITE_PER_TOKEN);
  });
});
