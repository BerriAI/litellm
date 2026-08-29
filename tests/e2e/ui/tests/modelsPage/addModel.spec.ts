import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_TEAM_CRUD_ID } from "../../constants";
import { Role, users } from "../../fixtures/users";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";
import { sendChatCompletion } from "../../helpers/traffic";

/** The mock LLM as the proxy reaches it: same host locally, a sidecar in the deployed stack. */
const MOCK_LLM_BASE = `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`;

/** GET /model/info?litellm_model_id= returns {data: [row]}, the deployment as stored. */
async function readDeployment(page: PlaywrightPage, modelId: string): Promise<Record<string, any> | undefined> {
  const body = await readBack<{ data: Record<string, any>[] }>(page, `/model/info?litellm_model_id=${modelId}`);
  return body.data[0];
}

/** GET /v2/model/info lists every deployment; created models are found by model_name. */
async function findDeploymentByName(page: PlaywrightPage, modelName: string): Promise<Record<string, any> | undefined> {
  const body = await readBack<{ data: Record<string, any>[] }>(page, "/v2/model/info");
  return body.data.find((row) => row.model_name === modelName);
}

/** Anchors a substring match to the whole string, escaping regex metacharacters. */
const exactly = (text: string): RegExp => new RegExp(`^${text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`);

/**
 * Helper to select a provider from the Add Model form dropdown. The field is a
 * searchable combobox: it only opens on click, typing filters the list, and the
 * option has to be picked explicitly because nothing is highlighted by default.
 * Options are matched on their visible text, not their accessible name, which
 * also carries the provider logo's alt text ("Anthropic logo Anthropic").
 */
async function selectProvider(page: PlaywrightPage, providerName: string) {
  const providerDropdown = page.getByRole("combobox", { name: "Provider", exact: true });
  await providerDropdown.click();
  await providerDropdown.fill(providerName);
  await page.getByRole("option").filter({ hasText: exactly(providerName) }).click();
  await expect(providerDropdown).toHaveValue(providerName);
}

test.describe("Add Model", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  // Set by the UI-add test below. The deployed stack keeps its database, so a leak
  // pollutes every later Models table and readback.
  let uiAddedModelName = "";

  test.afterEach(async ({ page }) => {
    if (!uiAddedModelName) return;
    const name = uiAddedModelName;
    uiAddedModelName = "";
    try {
      const stored = await findDeploymentByName(page, name);
      const id = stored?.model_info?.id;
      if (id) {
        await page.request.post("/model/delete", {
          headers: { Authorization: `Bearer ${users[Role.ProxyAdmin].password}` },
          data: { id },
        });
      }
    } catch {
      // Teardown must never turn a passing test red or mask a real failure.
    }
  });

  test("Able to see all models for a specific provider in the model dropdown", async ({ page }) => {
    await navigateToPage(page, Page.Models);
    await page.getByRole("tab", { name: "Add Model" }).click();

    await selectProvider(page, "Anthropic");

    // The model field should be a multi-select dropdown; click to open it
    await page.getByRole("combobox", { name: "Select models" }).click();

    // Verify provider-specific models are listed
    await expect(page.getByRole("option", { name: "claude-haiku-4-5", exact: true })).toBeVisible();
  });

  test("Edit team model TPM and RPM limits", async ({ page }) => {
    const masterKey = users[Role.ProxyAdmin].password;
    const modelName = `e2e-team-model-${Date.now()}`;

    // Create a team-scoped model via API so the test has something to edit.
    const createResponse = await page.request.post("/model/new", {
      headers: { Authorization: `Bearer ${masterKey}` },
      data: {
        model_name: modelName,
        litellm_params: {
          model: "openai/fake-gpt-4",
          // Never called, but the port moves when two checkouts run side by side.
          api_base: `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`,
          api_key: "fake-key",
          tpm: 100,
          rpm: 200,
        },
        model_info: {
          team_id: E2E_TEAM_CRUD_ID,
        },
      },
    });
    // A bare toBe(true) sends you looking at the UI for a setup call that never landed.
    expect(createResponse.ok(), `/model/new failed: ${createResponse.status()} ${await createResponse.text()}`).toBe(
      true,
    );
    const createdModelId = (await createResponse.json()).model_info?.id;
    expect(createdModelId, "model id from /model/new").toBeTruthy();

    // Navigate to Models + Endpoints
    await page.goto("/ui");
    await page.getByText("Models + Endpoints").click();

    // The Model ID cell is the drill-in control; the row itself is not clickable.
    const modelIdCell = page.getByTestId(`model-id-${createdModelId}`);
    await expect(modelIdCell).toBeVisible({ timeout: 10_000 });
    await modelIdCell.click();

    await expect(page.getByText("Back to Models").first()).toBeVisible({ timeout: 10_000 });

    // Edit Settings → change TPM/RPM → Save
    await page.getByRole("button", { name: "Edit Settings" }).click();

    await page.getByPlaceholder("Enter TPM").fill("999");
    await page.getByPlaceholder("Enter RPM").fill("888");

    // handleModelUpdate PATCHes the whole litellm_params blob, so pin what goes on the wire.
    const patch = await captureRequestBody(
      page,
      { method: "PATCH", urlIncludes: `/model/${createdModelId}/update` },
      async () => {
        await page.getByRole("button", { name: "Save Changes" }).click();
      },
    );
    expect(Number(patch.litellm_params?.tpm), "new TPM on the wire").toBe(999);
    expect(Number(patch.litellm_params?.rpm), "new RPM on the wire").toBe(888);

    // Verify the new values render back in view mode
    await expect(page.getByText("999", { exact: true })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("888", { exact: true })).toBeVisible({ timeout: 10_000 });

    // View mode re-renders from the form's own state, so read the deployment back.
    await expect
      .poll(
        async () => {
          const stored = await readDeployment(page, createdModelId);
          return [Number(stored?.litellm_params?.tpm), Number(stored?.litellm_params?.rpm)];
        },
        { message: "TPM/RPM did not persist on the deployment", timeout: 15_000 },
      )
      .toEqual([999, 888]);

    // Pin the fields this edit had no business changing; dropping them looks identical in the UI.
    const after = await readDeployment(page, createdModelId);
    expect(after?.litellm_params?.model, "upstream model untouched by a limits edit").toBe("openai/fake-gpt-4");
    expect(after?.model_info?.team_id, "team ownership untouched by a limits edit").toBe(E2E_TEAM_CRUD_ID);
  });

  test("Add a model through the UI, pass Test Connect, and serve traffic with it", async ({ page, request }) => {
    // Every other test here stops at "the row appears", which an unroutable model also does.
    // OpenAI-Compatible exposes API Base, so this points at the mock LLM and needs no credential.
    await navigateToPage(page, Page.Models);
    await page.getByRole("tab", { name: "Add Model" }).click();

    // Labels come from /public/providers/fields, not the frontend Providers enum, and the two differ.
    await selectProvider(page, "OpenAI-Compatible Endpoints (Together AI, etc.)");

    const publicName = `e2e-ui-added-${Date.now()}`;
    uiAddedModelName = publicName;

    // The model picker's "custom" entry reveals the free-text name field.
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: "Custom Model Name (Enter below)" }).click();
    await page.keyboard.press("Escape");
    await page.getByPlaceholder("Enter custom model name").fill(publicName);

    // By Form.Item id, not placeholder: placeholders change with the provider selection.
    await page.locator("#api_base").fill(MOCK_LLM_BASE);
    await page.locator("#api_key").fill("fake-key");

    await page.getByRole("button", { name: "Test Connect" }).click();
    await expect(page.getByText("Connection Test Results")).toBeVisible({ timeout: 10_000 });
    // Assert the success panel is present; "no failure yet" is also true mid-flight.
    await expect(page.getByTestId("connection-success-msg")).toBeVisible({ timeout: 30_000 });

    // The modal swallows the Add click. Scope to the footer: the dismiss X is also named "Close".
    const resultsModal = page.getByRole("dialog", { name: "Connection Test Results" });
    await resultsModal.locator('[data-slot="dialog-footer"]').getByRole("button", { name: "Close" }).click();
    await expect(resultsModal).toBeHidden({ timeout: 5_000 });

    const created = await captureRequestBody(page, { method: "POST", urlIncludes: "/model/new" }, async () => {
      await page.getByRole("button", { name: "Add Model" }).last().click();
    });
    expect(created.model_name, "the model is created under the name that was typed").toBe(publicName);
    expect(created.litellm_params?.api_base, "the api base survives the form").toBe(MOCK_LLM_BASE);

    await expect(page.getByText("created successfully")).toBeVisible({ timeout: 15_000 });

    // Serving one request is the only assertion that rules out a dropped api_base or an
    // unregistered name. Polled because /model/new returns before the router reloads.
    await expect
      .poll(
        async () => {
          try {
            await sendChatCompletion(request, { model: publicName, prompt: `hello from ${publicName}` });
            return true;
          } catch {
            return false;
          }
        },
        { message: `model ${publicName} was added through the UI but never served a request`, timeout: 30_000 },
      )
      .toBe(true);
  });

  test("Test connection with bad credentials shows failure", async ({ page }) => {
    await navigateToPage(page, Page.Models);
    await page.getByRole("tab", { name: "Add Model" }).click();

    await selectProvider(page, "Anthropic");

    // Select model: claude-haiku-4-5
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: "claude-haiku-4-5", exact: true }).click();
    await page.keyboard.press("Escape");

    // Enter bad API key
    const apiKeyInput = page.locator('input[type="password"]').first();
    await apiKeyInput.fill("sk-bad-key-12345");

    // Click Test Connect button by its text
    await page.getByRole("button", { name: "Test Connect" }).click();

    // Wait for modal to appear and connection test to complete
    await expect(page.getByText("Connection Test Results")).toBeVisible({ timeout: 10_000 });

    // Verify failure message appears (the test makes a real API call, so it will fail with bad creds)
    await expect(page.getByText(/Connection to .* failed/)).toBeVisible({ timeout: 30_000 });
  });

  test("Add specific model and verify it appears in All Models", async ({ page }) => {
    await navigateToPage(page, Page.Models);
    await page.getByRole("tab", { name: "Add Model" }).click();

    await selectProvider(page, "Anthropic");

    // Select model: claude-haiku-4-5
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: "claude-haiku-4-5", exact: true }).click();
    await page.keyboard.press("Escape");

    // Enter any API key
    const apiKeyInput = page.locator('input[type="password"]').first();
    await apiKeyInput.fill("sk-any-key-for-add-test");

    // Click Add Model button by its text
    const created = await captureRequestBody(page, { method: "POST", urlIncludes: "/model/new" }, async () => {
      await page.getByRole("button", { name: "Add Model" }).last().click();
    });
    // The form sends custom_llm_provider separately from the name, so both halves have to arrive.
    expect(created.model_name, "the selected model is what goes on the wire").toBe("claude-haiku-4-5");
    expect(created.litellm_params?.model, "the model name goes on the wire").toBe("claude-haiku-4-5");
    expect(created.litellm_params?.custom_llm_provider, "the picked provider goes on the wire").toBe("anthropic");

    // Wait for success notification
    await expect(page.getByText("created successfully")).toBeVisible({ timeout: 15_000 });

    // Navigate to All Models tab
    await page.getByRole("tab", { name: "All Models" }).click();
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(2000);

    // Search for the model we just added
    await page.getByPlaceholder("Search model names").fill("claude-haiku-4-5");
    await page.waitForTimeout(1000);

    // Verify the model appears in the results count (not "Showing 0 results")
    await expect(page.getByTestId("pagination-range")).toHaveText(/Showing \d+-\d+ of \d+/, {
      timeout: 15_000,
    });

    // Verify the model name appears in the table body
    const tableBody = page.locator("table tbody");
    await expect(tableBody.getByText("claude-haiku-4-5").first()).toBeVisible({ timeout: 15_000 });

    // A row proves the name is there, not what the deployment routes to.
    const stored = await findDeploymentByName(page, "claude-haiku-4-5");
    expect(stored, "created model readable from /v2/model/info").toBeTruthy();
    expect(stored?.litellm_params?.model, "stored deployment keeps the model name").toBe("claude-haiku-4-5");
    expect(stored?.litellm_params?.custom_llm_provider, "stored deployment keeps its provider").toBe("anthropic");
  });

  test("Add team-only model via Team-BYOK toggle and verify it appears with the team", async ({ page, request }) => {
    // The Team-BYOK switch is gated on premiumUser; without a license the toggle is disabled.
    test.skip(!process.env.LITELLM_LICENSE, "LITELLM_LICENSE not set in test env — Team-BYOK switch is disabled");

    // Idempotent across reruns. Only target rows with both the cohere name and the e2e team,
    // so the sibling wildcard test's team-less model is left alone.
    const masterKey = users[Role.ProxyAdmin].password;
    const auth = { Authorization: `Bearer ${masterKey}` };
    const deleteTeamScopedCohereModels = async () => {
      const res = await request.get("/v2/model/info", { headers: auth });
      if (!res.ok()) return;
      const body = await res.json();
      const matches: Array<{ id: string }> = (body?.data ?? []).filter(
        (m: any) =>
          typeof m?.model_name === "string" &&
          m.model_name.startsWith("cohere") &&
          m?.model_info?.team_id === E2E_TEAM_CRUD_ID,
      );
      for (const m of matches) {
        await request.post("/model/delete", { headers: auth, data: { id: m.id } });
      }
    };
    await deleteTeamScopedCohereModels();

    try {
      await navigateToPage(page, Page.Models);
      await page.getByRole("tab", { name: "Add Model" }).click();

      await selectProvider(page, "Cohere");

      await page.getByRole("combobox", { name: "Select models" }).click();
      await page.getByRole("option", { name: /All .* Models \(Wildcard\)/ }).click();
      await page.keyboard.press("Escape");

      const apiKeyInput = page.locator('input[type="password"]').first();
      await apiKeyInput.fill("sk-any-key-for-team-byok-test");

      // Flip the Team-BYOK switch on; the Switch carries its own aria-label.
      await page.getByRole("switch", { name: "Team-BYOK Model" }).click();

      // TeamDropdown options show the alias above the team id, so match on the id line by text.
      const teamDropdown = page.getByTestId("team-dropdown").getByRole("combobox");
      await expect(teamDropdown).toBeVisible({ timeout: 5_000 });
      await teamDropdown.click();
      const teamOption = page.locator('[data-slot="combobox-content"]:visible').getByText(E2E_TEAM_CRUD_ID).first();
      await expect(teamOption).toBeVisible({ timeout: 5_000 });
      await teamOption.click();

      await page.getByRole("button", { name: "Add Model" }).last().click();

      // Scope to the toast container so a stale toast can't satisfy this.
      await expect(page.locator("[data-sonner-toast]").getByText("created successfully").last()).toBeVisible({
        timeout: 15_000,
      });

      // The Models table renders team-scoped models with the team id in the row.
      await page.getByRole("tab", { name: "All Models" }).click();
      await page.waitForLoadState("networkidle");
      // networkidle fires before the table finishes re-rendering.
      await page.waitForTimeout(2000);

      await page.getByPlaceholder("Search model names").fill("cohere");
      await page.waitForTimeout(1000);

      // Clearer failure than timing out on a row assertion when the table is empty.
      await expect(page.getByTestId("pagination-range")).toHaveText(/Showing \d+-\d+ of \d+/, {
        timeout: 15_000,
      });

      // Pin to one row carrying both the name and the team, so the sibling test's
      // team-less cohere row can't satisfy it.
      const teamCohereRow = page
        .locator("table tbody tr")
        .filter({ hasText: "cohere/" })
        .filter({ hasText: E2E_TEAM_CRUD_ID });
      await expect(teamCohereRow).toHaveCount(1, { timeout: 15_000 });
    } finally {
      await deleteTeamScopedCohereModels();
    }
  });

  test("Add wildcard route and verify it appears in All Models", async ({ page }) => {
    await navigateToPage(page, Page.Models);
    await page.getByRole("tab", { name: "Add Model" }).click();

    await selectProvider(page, "Cohere");

    // Select All Cohere Models (Wildcard)
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: /All .* Models \(Wildcard\)/ }).click();
    await page.keyboard.press("Escape");

    // Enter any API key
    const apiKeyInput = page.locator('input[type="password"]').first();
    await apiKeyInput.fill("sk-any-key-for-wildcard-test");

    // Click Add Model button by its text
    const created = await captureRequestBody(page, { method: "POST", urlIncludes: "/model/new" }, async () => {
      await page.getByRole("button", { name: "Add Model" }).last().click();
    });
    // A wildcard with the star stripped becomes a plain "cohere" deployment that matches nothing.
    expect(created.model_name, "the wildcard route goes on the wire intact").toBe("cohere/*");

    // Wait for success notification
    await expect(page.getByText("created successfully")).toBeVisible({ timeout: 15_000 });

    // Navigate to All Models tab
    await page.getByRole("tab", { name: "All Models" }).click();
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(2000);

    // Search for the wildcard model
    await page.getByPlaceholder("Search model names").fill("cohere");
    await page.waitForTimeout(1000);

    // Verify the model appears in the results count (not "Showing 0 results")
    await expect(page.getByTestId("pagination-range")).toHaveText(/Showing \d+-\d+ of \d+/, {
      timeout: 15_000,
    });

    // Verify the wildcard model appears in the table body (wildcard models show as "cohere/*")
    const tableBody = page.locator("table tbody");
    await expect(tableBody.getByText("cohere/").first()).toBeVisible({ timeout: 15_000 });

    // "cohere/" in the table also matches a plain cohere deployment; require the wildcard exactly.
    const stored = await findDeploymentByName(page, "cohere/*");
    expect(stored, "wildcard deployment readable from /v2/model/info").toBeTruthy();
    expect(stored?.litellm_params?.model, "stored deployment keeps the wildcard route").toBe("cohere/*");
  });
});
