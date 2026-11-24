import { test, expect, APIRequestContext, Page as PlaywrightPage } from "@playwright/test";
import { randomUUID } from "node:crypto";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { captureRequestBody } from "../../helpers/roundTrip";

const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
const headers = { Authorization: `Bearer ${master}` };
const upstream = (process.env.INTEGRATION_UPSTREAM_URL ?? "http://127.0.0.1:8190").replace(/\/+$/, "");
const CORALBRICKS_MODEL = "deepseek-v4.1-flash-fast";
const UI_KEY = "cb_ui_integration_key";
const UPSTREAM_PROMPT_TOKENS = 20;
const UPSTREAM_COMPLETION_TOKENS = 20;

type DeploymentRow = {
  model_name: string;
  litellm_params?: { model?: string; custom_llm_provider?: string };
  model_info?: { id?: string };
};

type CatalogRow = { input_cost_per_token?: number; output_cost_per_token?: number };

type SpendRow = {
  request_id: string;
  custom_llm_provider: string;
  model: string;
  model_group: string;
  prompt_tokens: number;
  completion_tokens: number;
  spend: number | string;
};

const optionNamed = (label: string): RegExp =>
  new RegExp(`(^|\\s)${label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`);

async function loginAsAdmin(page: PlaywrightPage): Promise<void> {
  await page.goto("/ui/login");
  await page.getByPlaceholder("Enter your username").fill("admin");
  await page.getByPlaceholder("Enter your password").fill(master);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await expect(page).toHaveURL(
    (url) => url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
  );
}

async function selectProvider(page: PlaywrightPage, providerName: string): Promise<void> {
  const providerDropdown = page.getByRole("combobox", { name: "Provider", exact: true });
  await providerDropdown.click();
  await providerDropdown.fill(providerName);
  await page.getByRole("option", { name: optionNamed(providerName) }).click();
  await expect(providerDropdown).toHaveValue(providerName);
}

async function deploymentNamed(
  request: APIRequestContext,
  modelName: string,
): Promise<DeploymentRow | undefined> {
  const response = await request.get("/v1/model/info", { headers });
  expect(response.ok(), await response.text()).toBe(true);
  const rows = (await response.json()).data as DeploymentRow[];
  return rows.find((row) => row.model_name === modelName);
}

async function deleteDeployment(request: APIRequestContext, modelName: string): Promise<void> {
  const stored = await deploymentNamed(request, modelName);
  const id = stored?.model_info?.id;
  if (!id) return;
  const deleted = await request.post("/model/delete", { headers, data: { id } });
  expect(deleted.ok(), await deleted.text()).toBe(true);
}

async function chatOnce(request: APIRequestContext, modelName: string): Promise<{ status: number; body: string }> {
  const response = await request.post("/v1/chat/completions", {
    headers,
    data: { model: modelName, messages: [{ role: "user", content: `hello from ${modelName}` }] },
  });
  return { status: response.status(), body: await response.text() };
}

async function spendRow(request: APIRequestContext, requestId: string): Promise<SpendRow | undefined> {
  const response = await request.get(`/spend/logs?request_id=${encodeURIComponent(requestId)}`, { headers });
  expect(response.ok(), await response.text()).toBe(true);
  const rows = (await response.json()) as SpendRow[];
  return rows[0];
}

test("the Add Model form offers CoralBricks and the deployment it creates serves and bills a chat", async ({
  page,
  request,
}) => {
  const alias = `integration-ui-coralbricks-${randomUUID().replace(/-/g, "").slice(0, 12)}`;
  try {
    await loginAsAdmin(page);
    await navigateToPage(page, Page.Models);
    const catalogLoaded = page.waitForResponse(
      (response) => response.url().includes("/public/litellm_model_cost_map?catalog_only=true") && response.ok(),
    );
    await page.getByRole("tab", { name: "Add Model" }).click();
    const catalog = (await (await catalogLoaded).json()) as Record<string, CatalogRow>;
    const catalogRow = catalog[`coralbricks/${CORALBRICKS_MODEL}`];

    await selectProvider(page, "CoralBricks");

    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: "Custom Model Name (Enter below)" }).click();
    await page.keyboard.press("Escape");
    await page.getByPlaceholder("Enter custom model name").fill(CORALBRICKS_MODEL);

    const publicName = page.getByRole("row").filter({ hasText: CORALBRICKS_MODEL }).getByRole("textbox");
    await expect(publicName).toHaveValue(CORALBRICKS_MODEL);
    await publicName.fill(alias);

    await page.locator("#api_base").fill(`${upstream}/v1`);
    await page.locator("#api_key").fill(UI_KEY);

    const created = await captureRequestBody(page, { method: "POST", urlIncludes: "/model/new" }, async () => {
      await page.getByTestId("add-model-btn").click();
    });
    expect(created.model_name, "the public name typed into the mapping row").toBe(alias);
    expect(created.litellm_params?.model, "the custom model name typed into the form").toBe(CORALBRICKS_MODEL);
    expect(created.litellm_params?.custom_llm_provider, "the provider the picker maps CoralBricks to").toBe(
      "coralbricks",
    );
    expect(created.litellm_params?.api_base).toBe(`${upstream}/v1`);
    expect(created.litellm_params?.api_key).toBe(UI_KEY);
    await expect(page.getByText("created successfully")).toBeVisible({ timeout: 15_000 });

    let served = { status: 0, body: "" };
    await expect
      .poll(
        async () => {
          served = await chatOnce(request, alias);
          return served.status;
        },
        { message: `the deployment added through the UI never served a chat: ${served.body}`, timeout: 60_000 },
      )
      .toBe(200);
    const requestId = JSON.parse(served.body).id as string;
    expect(requestId, served.body).toBeTruthy();

    await expect
      .poll(async () => (await deploymentNamed(request, alias))?.litellm_params?.custom_llm_provider, {
        message: `no worker listed ${alias}`,
        timeout: 60_000,
      })
      .toBe("coralbricks");
    const inputRate = catalogRow?.input_cost_per_token;
    const outputRate = catalogRow?.output_cost_per_token;
    expect(inputRate, JSON.stringify(catalogRow)).toBeGreaterThan(0);
    expect(outputRate, JSON.stringify(catalogRow)).toBeGreaterThan(0);
    const expectedSpend = UPSTREAM_PROMPT_TOKENS * inputRate! + UPSTREAM_COMPLETION_TOKENS * outputRate!;

    let row: SpendRow | undefined;
    await expect
      .poll(
        async () => {
          row = await spendRow(request, requestId);
          return row?.request_id;
        },
        { message: `no spend row for ${requestId}`, timeout: 60_000 },
      )
      .toBe(requestId);
    expect(row?.custom_llm_provider, JSON.stringify(row)).toBe("coralbricks");
    expect(row?.model, JSON.stringify(row)).toBe(created.litellm_params.model);
    expect(row?.model_group, JSON.stringify(row)).toBe(alias);
    expect(row?.prompt_tokens, JSON.stringify(row)).toBe(UPSTREAM_PROMPT_TOKENS);
    expect(row?.completion_tokens, JSON.stringify(row)).toBe(UPSTREAM_COMPLETION_TOKENS);
    expect(Number(row?.spend), JSON.stringify(row)).toBeCloseTo(expectedSpend, 10);
  } finally {
    await deleteDeployment(request, alias);
  }
});
