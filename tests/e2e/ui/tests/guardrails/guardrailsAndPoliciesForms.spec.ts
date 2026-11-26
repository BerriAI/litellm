import { expect, test, type Page } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { masterKey, rootPath } from "../../helpers/traffic";

interface StoredGuardrail {
  guardrail_id: string;
  guardrail_name: string | null;
}

interface StoredPolicy {
  policy_id: string;
  policy_name: string;
}

async function deleteGuardrailByName(page: Page, name: string): Promise<void> {
  const res = await page.request.get(`${rootPath()}/v2/guardrails/list`, {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  if (!res.ok()) return;
  const rows = ((await res.json()) as { guardrails?: StoredGuardrail[] }).guardrails ?? [];
  for (const row of rows.filter((g) => g.guardrail_name === name)) {
    await page.request.delete(`${rootPath()}/guardrails/${row.guardrail_id}`, {
      headers: { Authorization: `Bearer ${masterKey()}` },
    });
  }
}

async function deletePolicyByName(page: Page, name: string): Promise<void> {
  const res = await page.request.get(`${rootPath()}/policies/list`, {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  if (!res.ok()) return;
  const rows = ((await res.json()) as { policies?: StoredPolicy[] }).policies ?? [];
  for (const row of rows.filter((p) => p.policy_name === name)) {
    await page.request.delete(`${rootPath()}/policies/${row.policy_id}`, {
      headers: { Authorization: `Bearer ${masterKey()}` },
    });
  }
}

test.describe("Guardrails and policies forms", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  const createdGuardrailNames: string[] = [];
  const createdPolicyNames: string[] = [];

  test.afterEach(async ({ page }) => {
    for (const name of createdPolicyNames.splice(0)) {
      await deletePolicyByName(page, name);
    }
    for (const name of createdGuardrailNames.splice(0)) {
      await deleteGuardrailByName(page, name);
    }
  });

  test("guardrail create and edit validate empty names and send expected payloads", async ({ page }) => {
    const suffix = `${Date.now()}-${Math.floor(Math.random() * 10000)}`;
    const initialName = `e2e-g14-cf-${suffix}`;
    const renamedName = `${initialName}-edited`;
    createdGuardrailNames.push(initialName, renamedName);

    const postRequests: string[] = [];
    page.on("request", (req) => {
      if (req.method() === "POST" && new URL(req.url()).pathname.endsWith("/guardrails")) {
        postRequests.push(req.url());
      }
    });

    await page.goto(`${rootPath()}/ui/guardrails`, { waitUntil: "networkidle" });
    await page.getByRole("button", { name: "Add New Guardrail" }).click();
    await page.getByRole("menuitem", { name: "Add Provider Guardrail" }).click();

    const wizard = page.getByRole("dialog", { name: "Create guardrail" });
    await expect(wizard).toBeVisible({ timeout: 10_000 });

    await wizard.getByRole("button", { name: "Next" }).click();
    await expect(wizard.getByText("Please enter a guardrail name")).toBeVisible({ timeout: 5_000 });
    expect(postRequests).toEqual([]);

    await wizard.getByRole("textbox", { name: "Guardrail Name" }).fill(initialName);
    await wizard.getByRole("combobox", { name: "Guardrail Provider" }).click();
    await page.getByRole("option", { name: /LiteLLM Content Filter/ }).click();

    for (const step of ["Topics", "Patterns", "Keywords"]) {
      await wizard.getByRole("button", { name: "Next" }).click();
      await expect(wizard).toContainText(step, { timeout: 10_000 });
    }

    await wizard.getByRole("button", { name: "Add Keyword" }).click();
    const keywordDialog = page.locator('[role="dialog"]').last();
    await keywordDialog.locator("input").first().fill("e2e-blocked-term");
    await keywordDialog.getByRole("button", { name: "Add", exact: true }).click();
    await wizard.getByRole("button", { name: "Next" }).click();

    const createResponsePromise = page.waitForResponse(
      (res) => res.request().method() === "POST" && new URL(res.url()).pathname.endsWith("/guardrails"),
    );
    await wizard.getByRole("button", { name: "Create Guardrail" }).click();
    const createResponse = await createResponsePromise;
    expect(createResponse.ok(), `POST /guardrails (${createResponse.status()})`).toBe(true);
    const createBody = createResponse.request().postDataJSON();
    expect(createBody.guardrail.guardrail_name).toBe(initialName);
    expect(createBody.guardrail.litellm_params.guardrail).toBe("litellm_content_filter");
    expect(createBody.guardrail.litellm_params.blocked_words).toEqual(
      expect.arrayContaining([expect.objectContaining({ keyword: "e2e-blocked-term" })]),
    );
    await expect(wizard).toBeHidden({ timeout: 10_000 });

    const providerParamsLoaded = page.waitForResponse((res) =>
      res.url().includes("/guardrails/ui/provider_specific_params"),
    );
    await page.getByRole("row").filter({ hasText: initialName }).getByRole("button").first().click();
    await expect(page.getByRole("button", { name: "Back to Guardrails" })).toBeVisible({ timeout: 10_000 });
    await providerParamsLoaded;
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();

    const patchRequests: string[] = [];
    page.on("request", (req) => {
      if (req.method() === "PATCH" && new URL(req.url()).pathname.includes("/guardrails/")) {
        patchRequests.push(req.url());
      }
    });

    const editNameInput = page.getByPlaceholder("Enter guardrail name");
    await editNameInput.fill("");
    await page.getByRole("button", { name: "Save Changes" }).click();
    await expect(page.getByText("Please input a guardrail name")).toBeVisible({ timeout: 5_000 });
    expect(patchRequests).toEqual([]);

    await editNameInput.fill(renamedName);
    const patchResponsePromise = page.waitForResponse(
      (res) => res.request().method() === "PATCH" && new URL(res.url()).pathname.includes("/guardrails/"),
    );
    await page.getByRole("button", { name: "Save Changes" }).click();
    const patchResponse = await patchResponsePromise;
    expect(patchResponse.ok(), `PATCH /guardrails/{id} (${patchResponse.status()})`).toBe(true);
    const patchBody = patchResponse.request().postDataJSON();
    expect(patchBody.guardrail_name).toBe(renamedName);
  });

  test("custom code guardrail creates and updates from guardrail info", async ({ page }) => {
    const suffix = `${Date.now()}-${Math.floor(Math.random() * 10000)}`;
    const initialName = `e2e-g14-code-${suffix}`;
    const renamedName = `${initialName}-edited`;
    createdGuardrailNames.push(initialName, renamedName);

    await page.goto(`${rootPath()}/ui/guardrails`, { waitUntil: "networkidle" });
    await page.getByRole("button", { name: "Add New Guardrail" }).click();
    await page.getByRole("menuitem", { name: "Create Custom Code Guardrail" }).click();

    const codeDialog = page.locator('[role="dialog"]').first();
    await expect(codeDialog).toBeVisible({ timeout: 10_000 });
    await expect(codeDialog.getByRole("button", { name: "Save Guardrail" })).toBeDisabled();

    await codeDialog.getByPlaceholder("e.g., block-pii-custom").fill(initialName);
    const createPromise = page.waitForResponse(
      (res) => res.request().method() === "POST" && new URL(res.url()).pathname.endsWith("/guardrails"),
    );
    await codeDialog.getByRole("button", { name: "Save Guardrail" }).click();
    const createRes = await createPromise;
    expect(createRes.ok(), `POST /guardrails custom_code (${createRes.status()})`).toBe(true);
    expect(createRes.request().postDataJSON().guardrail.litellm_params.guardrail).toBe("custom_code");
    await expect(codeDialog).toBeHidden({ timeout: 10_000 });

    const infoSettled = Promise.all([
      page.waitForResponse((res) => res.url().includes("/guardrails/ui/provider_specific_params")),
      page.waitForResponse((res) => res.url().includes("/guardrails/ui/add_guardrail_settings")),
    ]);
    await page.getByRole("row").filter({ hasText: initialName }).getByRole("button").first().click();
    await expect(page.getByRole("button", { name: "Back to Guardrails" })).toBeVisible({ timeout: 10_000 });
    await infoSettled;
    await page.getByRole("button", { name: "Edit Code" }).click();
    await expect(codeDialog).toBeVisible({ timeout: 10_000 });

    const editCodeNameInput = codeDialog.getByPlaceholder("e.g., block-pii-custom");
    await expect(editCodeNameInput).toHaveValue(initialName);
    await editCodeNameInput.fill(renamedName);
    const updatePromise = page.waitForResponse(
      (res) => res.request().method() === "PATCH" && new URL(res.url()).pathname.includes("/guardrails/"),
    );
    await codeDialog.getByRole("button", { name: "Update Guardrail" }).click();
    const updateRes = await updatePromise;
    expect(updateRes.ok(), `PATCH /guardrails/{id} custom_code (${updateRes.status()})`).toBe(true);
    expect(updateRes.request().postDataJSON().guardrail_name).toBe(renamedName);
    await expect(codeDialog).toBeHidden({ timeout: 10_000 });
  });

  test("policy pipeline validates empty name and saves pipeline payload", async ({ page }) => {
    const suffix = `${Date.now()}-${Math.floor(Math.random() * 10000)}`;
    const guardrailName = `e2e-g14-pol-gr-${suffix}`;
    const policyName = `e2e-g14-policy-${suffix}`;
    createdGuardrailNames.push(guardrailName);
    createdPolicyNames.push(policyName);

    const grRes = await page.request.post(`${rootPath()}/guardrails`, {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: {
        guardrail: {
          guardrail_name: guardrailName,
          litellm_params: {
            guardrail: "litellm_content_filter",
            mode: "pre_call",
            default_on: false,
            blocked_words: [{ keyword: "policy-seed-word", action: "BLOCK" }],
          },
        },
      },
    });
    expect(grRes.ok(), `Seed guardrail (${grRes.status()})`).toBe(true);

    const postRequests: string[] = [];
    page.on("request", (req) => {
      if (req.method() === "POST" && new URL(req.url()).pathname.endsWith("/policies")) {
        postRequests.push(req.url());
      }
    });

    await page.goto(`${rootPath()}/ui/policies`, { waitUntil: "networkidle" });
    await page.getByRole("tab", { name: "Policies", exact: true }).click();
    await page.getByRole("button", { name: "Add New Policy" }).click();

    const modeDialog = page.locator('[role="dialog"]').first();
    await expect(modeDialog).toBeVisible({ timeout: 10_000 });
    await modeDialog.getByText("Flow Builder").click();
    await modeDialog.getByRole("button", { name: "Continue to Builder" }).click();

    const policyNameInput = page.getByPlaceholder("Policy name...");
    await expect(policyNameInput).toBeVisible({ timeout: 10_000 });

    await page.getByRole("button", { name: "Save Policy" }).click();
    await expect(page.getByText("Please enter a policy name")).toBeVisible({ timeout: 5_000 });
    expect(postRequests).toEqual([]);

    await policyNameInput.fill(policyName);
    await page.getByPlaceholder("Add a description (optional)...").fill("E2E pipeline description");
    await page.getByRole("combobox", { name: "Select a guardrail" }).first().click();
    await page.getByRole("option", { name: guardrailName }).first().click();

    const createPolicyPromise = page.waitForResponse(
      (res) => res.request().method() === "POST" && new URL(res.url()).pathname.endsWith("/policies"),
    );
    await page.getByRole("button", { name: "Save Policy" }).click();
    const createPolicyRes = await createPolicyPromise;
    expect(createPolicyRes.ok(), `POST /policies (${createPolicyRes.status()})`).toBe(true);
    const body = createPolicyRes.request().postDataJSON();
    expect(body.policy_name).toBe(policyName);
    expect(body.description).toBe("E2E pipeline description");
    expect(body.guardrails_add).toEqual([guardrailName]);
    expect(body.pipeline.steps).toHaveLength(1);
    expect(body.pipeline.steps[0].guardrail).toBe(guardrailName);
  });
});
