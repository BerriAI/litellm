import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH, E2E_TEAM_CRUD_ID } from "../../constants";
import { Role, users } from "../../fixtures/users";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";

/** GET /model/info?litellm_model_id= returns {data: [row]} — the deployment as stored. */
async function readDeployment(page: PlaywrightPage, modelId: string): Promise<Record<string, any> | undefined> {
  const body = await readBack<{ data: Record<string, any>[] }>(page, `/model/info?litellm_model_id=${modelId}`);
  return body.data[0];
}

/** GET /v2/model/info lists every deployment; created models are found by model_name. */
async function findDeploymentByName(page: PlaywrightPage, modelName: string): Promise<Record<string, any> | undefined> {
  const body = await readBack<{ data: Record<string, any>[] }>(page, "/v2/model/info");
  return body.data.find((row) => row.model_name === modelName);
}

/**
 * Helper to select a provider from the Add Model form dropdown.
 */
async function selectProvider(page: any, providerName: string) {
  const providerDropdown = page.getByRole("combobox", { name: /Provider/i });
  await providerDropdown.fill(providerName);
  await page.waitForTimeout(1000);
  await providerDropdown.press("Enter");
  await page.waitForTimeout(2000);
}

test.describe("Add Model", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Able to see all models for a specific provider in the model dropdown", async ({ page }) => {
    await navigateToPage(page, Page.Models);
    await page.getByRole("tab", { name: "Add Model" }).click();

    await selectProvider(page, "Anthropic");

    // The model field should be a multi-select dropdown; click to open it
    const modelDropdown = page.locator(".ant-select-selection-overflow").first();
    await modelDropdown.click();

    // Verify provider-specific models are listed
    await expect(page.getByTitle("claude-haiku-4-5", { exact: true })).toBeVisible();
  });

  test("Edit team model TPM and RPM limits", async ({ page }) => {
    const masterKey = users[Role.ProxyAdmin].password;
    const modelName = `e2e-team-model-${Date.now()}`;

    // Create a team-scoped model via API so the test has something to edit.
    // The e2e runner spins up a fresh postgres container per invocation, so
    // there's no cleanup step — the DB is thrown away at the end of the run.
    const createResponse = await page.request.post("/model/new", {
      headers: { Authorization: `Bearer ${masterKey}` },
      data: {
        model_name: modelName,
        litellm_params: {
          model: "openai/fake-gpt-4",
          // Nothing here ever calls the model, but point it at the mock the
          // harness actually started -- the port moves when two checkouts run
          // the stack side by side.
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
    // Say why it failed. A bare toBe(true) here reports "expected true, received
    // false" and sends you looking at the UI for a setup call that never landed.
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

    // handleModelUpdate rebuilds the whole litellm_params blob on every save and
    // PATCHes it wholesale, so what goes on the wire is worth pinning: the edit
    // is only correct if the two changed fields are in there.
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

    // ...but view mode re-renders from the form's own state, so it shows what the
    // UI asked for whether or not the backend kept it. Read the deployment back.
    await expect
      .poll(
        async () => {
          const stored = await readDeployment(page, createdModelId);
          return [Number(stored?.litellm_params?.tpm), Number(stored?.litellm_params?.rpm)];
        },
        { message: "TPM/RPM did not persist on the deployment", timeout: 15_000 },
      )
      .toEqual([999, 888]);

    // A save that lands the edited fields and silently drops the untouched ones
    // is the exact shape of the reported regressions, and it looks identical in
    // the UI. Pin the fields this edit had no business changing.
    const after = await readDeployment(page, createdModelId);
    expect(after?.litellm_params?.model, "upstream model untouched by a limits edit").toBe("openai/fake-gpt-4");
    expect(after?.model_info?.team_id, "team ownership untouched by a limits edit").toBe(E2E_TEAM_CRUD_ID);
  });

  test("Test connection with bad credentials shows failure", async ({ page }) => {
    await navigateToPage(page, Page.Models);
    await page.getByRole("tab", { name: "Add Model" }).click();

    await selectProvider(page, "Anthropic");

    // Select model: claude-haiku-4-5
    const modelDropdown = page.locator(".ant-select-selection-overflow").first();
    await modelDropdown.click();
    await page.getByTitle("claude-haiku-4-5", { exact: true }).click();
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
    const modelDropdown = page.locator(".ant-select-selection-overflow").first();
    await modelDropdown.click();
    await page.getByTitle("claude-haiku-4-5", { exact: true }).click();
    await page.keyboard.press("Escape");

    // Enter any API key
    const apiKeyInput = page.locator('input[type="password"]').first();
    await apiKeyInput.fill("sk-any-key-for-add-test");

    // Click Add Model button by its text
    const created = await captureRequestBody(page, { method: "POST", urlIncludes: "/model/new" }, async () => {
      await page.getByRole("button", { name: "Add Model" }).last().click();
    });
    // The form carries the provider separately from the model name (it sends
    // custom_llm_provider rather than an "anthropic/" prefix), so both halves
    // have to arrive. A deployment that loses the provider selection looks
    // right in the table and is unroutable.
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

    // The table is populated from the model list the UI refetches, but a row is
    // only evidence the name is there -- it says nothing about what the
    // deployment actually routes to. Read the stored deployment back.
    const stored = await findDeploymentByName(page, "claude-haiku-4-5");
    expect(stored, "created model readable from /v2/model/info").toBeTruthy();
    expect(stored?.litellm_params?.model, "stored deployment keeps the model name").toBe("claude-haiku-4-5");
    expect(stored?.litellm_params?.custom_llm_provider, "stored deployment keeps its provider").toBe("anthropic");
  });

  test("Add team-only model via Team-BYOK toggle and verify it appears with the team", async ({ page, request }) => {
    // The Team-BYOK switch is gated on `premiumUser` — without a license set
    // for the proxy under test, the toggle is disabled and this manual-QA
    // step cannot be exercised.
    test.skip(!process.env.LITELLM_LICENSE, "LITELLM_LICENSE not set in test env — Team-BYOK switch is disabled");

    // Make the test idempotent across retries and local reruns: delete any
    // Cohere model already scoped to the e2e team before we start, and again
    // after we finish. The sibling "Add wildcard route" test creates a
    // team-less Cohere wildcard, so we only target rows that have BOTH the
    // cohere/* model_name AND team_id == e2e-team-crud.
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

      const modelDropdown = page.locator(".ant-select-selection-overflow").first();
      await modelDropdown.click();
      const wildcardOption = page.getByTitle(/All .* Models \(Wildcard\)/);
      await wildcardOption.click();
      await page.keyboard.press("Escape");

      const apiKeyInput = page.locator('input[type="password"]').first();
      await apiKeyInput.fill("sk-any-key-for-team-byok-test");

      // Flip the Team-BYOK switch on (Form.Item label "Team-BYOK Model")
      const teamByokRow = page.locator(".ant-form-item", { hasText: "Team-BYOK Model" });
      await teamByokRow.getByRole("switch").click();

      // The Team dropdown appears underneath once the switch is on. TeamDropdown
      // renders its Select.Option children with custom <span>/<Text> markup, so
      // the popup items don't carry role="option" — match by text content,
      // scoped to the visible dropdown so a stale tag elsewhere in the form
      // can't satisfy it.
      const teamDropdown = page.getByTestId("team-dropdown");
      await expect(teamDropdown).toBeVisible({ timeout: 5_000 });
      await teamDropdown.click();
      const teamOption = page.locator(".ant-select-dropdown:visible").getByText(E2E_TEAM_CRUD_ID).first();
      await expect(teamOption).toBeVisible({ timeout: 5_000 });
      await teamOption.click();

      await page.getByRole("button", { name: "Add Model" }).last().click();

      // Scope the success toast to antd's notification container so a stale
      // success message from an earlier test in the same context can't satisfy
      // the assertion.
      await expect(page.locator(".ant-notification").getByText("created successfully").last()).toBeVisible({
        timeout: 15_000,
      });

      // Verify the model is now in All Models with the team_id attached. The
      // Models table renders team-scoped models with the team id in the row.
      await page.getByRole("tab", { name: "All Models" }).click();
      await page.waitForLoadState("networkidle");
      // Match the sibling tests in this file — networkidle fires before the
      // table finishes re-rendering, so give it the same 2s settle before
      // searching.
      await page.waitForTimeout(2000);

      await page.getByPlaceholder("Search model names").fill("cohere");
      await page.waitForTimeout(1000);

      // Confirm the search returned at least one result — gives a clear
      // failure message when the table is empty instead of timing out on a
      // row assertion.
      await expect(page.getByTestId("pagination-range")).toHaveText(/Showing \d+-\d+ of \d+/, {
        timeout: 15_000,
      });

      // Stronger than "the team appears somewhere in tbody" — pin the assertion
      // to a single row that has BOTH the cohere model_name AND the seeded
      // team, so a stale cohere row from "Add wildcard route" (no team) can't
      // satisfy the check. The Team ID column renders the id, not the alias.
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
    const modelDropdown = page.locator(".ant-select-selection-overflow").first();
    await modelDropdown.click();
    const wildcardOption = page.getByTitle(/All .* Models \(Wildcard\)/);
    await wildcardOption.click();
    await page.keyboard.press("Escape");

    // Enter any API key
    const apiKeyInput = page.locator('input[type="password"]').first();
    await apiKeyInput.fill("sk-any-key-for-wildcard-test");

    // Click Add Model button by its text
    const created = await captureRequestBody(page, { method: "POST", urlIncludes: "/model/new" }, async () => {
      await page.getByRole("button", { name: "Add Model" }).last().click();
    });
    // A wildcard that arrives with the star stripped becomes an ordinary
    // deployment named "cohere" and silently stops matching anything.
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

    // "cohere/" in the table would also match a plain cohere deployment. Read
    // the stored route back and require the wildcard exactly.
    const stored = await findDeploymentByName(page, "cohere/*");
    expect(stored, "wildcard deployment readable from /v2/model/info").toBeTruthy();
    expect(stored?.litellm_params?.model, "stored deployment keeps the wildcard route").toBe("cohere/*");
  });
});
