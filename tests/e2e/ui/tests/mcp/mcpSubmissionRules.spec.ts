import { test as base, expect, type APIRequestContext } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { deleteMcpServerByName } from "../../helpers/mcp";
import { navigateToPage } from "../../helpers/navigation";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";
import { Page } from "../../fixtures/pages";

const localServerUrl = process.env.MOCK_LLM_URL ?? "http://127.0.0.1:8090/v1";
const generalSettings = `${rootPath()}/config/list?config_type=general_settings`;

type Setting = { field_name: string; field_value: unknown; stored_in_db: boolean | null };
type AdminHeaders = { Authorization: string; "Content-Type": string };

async function readSetting(request: APIRequestContext, headers: AdminHeaders, field: string): Promise<Setting> {
  const response = await request.get(generalSettings, { headers: { ...headers, Connection: "close" } });
  try {
    await expect(response).toBeOK();
    const rows: Setting[] = await response.json();
    const row = rows.find(({ field_name }) => field_name === field);
    if (!row) throw new Error(`Missing general setting: ${field}`);
    return row;
  } finally {
    await response.dispose();
  }
}

async function waitForSetting(request: APIRequestContext, headers: AdminHeaders, expected: Setting) {
  // One fresh connection can hit the writer before the other workers catch up.
  let matching = 0;
  await expect
    .poll(async () => {
      const current = await readSetting(request, headers, expected.field_name);
      matching = current.stored_in_db === expected.stored_in_db &&
        JSON.stringify(current.field_value) === JSON.stringify(expected.field_value)
        ? matching + 1
        : 0;
      return matching;
    }, {
      message: `Wait for ${expected.field_name} to settle across proxy workers`,
      timeout: 60_000,
      intervals: [250],
    })
    .toBe(12);
}

async function withSetting<T>(
  request: APIRequestContext,
  headers: AdminHeaders,
  field: string,
  value: T,
  use: (value: T) => Promise<void>,
) {
  const previous = await readSetting(request, headers, field);
  const data = { field_name: field, config_type: "general_settings" };
  const seed = await request.post(`${rootPath()}/config/field/update`, {
    headers,
    data: { ...data, field_value: value },
  });
  await expect(seed).toBeOK();
  try {
    await waitForSetting(request, headers, { field_name: field, field_value: value, stored_in_db: true });
    await use(value);
  } finally {
    const restore = previous.stored_in_db
      ? await request.post(`${rootPath()}/config/field/update`, { headers, data: { ...data, field_value: previous.field_value } })
      : await request.post(`${rootPath()}/config/field/delete`, { headers, data });
    await expect(restore).toBeOK();
    await waitForSetting(request, headers, previous);
  }
}

interface Fixtures {
  adminHeaders: AdminHeaders;
  requiredFields: string[];
  submissionRules: string[];
  networkHeader: string;
  teamKey: string;
  submittedServerName: string;
}

const test = base.extend<Fixtures>({
  adminHeaders: async ({}, use) => {
    await use({
      Authorization: `Bearer ${masterKey()}`,
      "Content-Type": "application/json",
    });
  },
  requiredFields: [["description", "url", "alias"], { option: true }],
  submissionRules: async ({ request, adminHeaders, requiredFields }, use) => {
    await withSetting(request, adminHeaders, "mcp_required_fields", requiredFields, use);
  },
  networkHeader: async ({ request, adminHeaders }, use) => {
    await withSetting(request, adminHeaders, "mcp_client_id_header", `x-mcp-seed-${uniqueSuffix()}`, use);
  },
  teamKey: async ({ adminHeaders, page }, use) => {
    const keyResponse = await page.request.post(`${rootPath()}/key/generate`, {
      headers: adminHeaders,
      data: { team_id: "e2e-team-crud", user_id: "e2e-internal-user" },
    });
    expect(keyResponse.status(), await keyResponse.text()).toBe(200);
    const key = (await keyResponse.json()) as { key: string; token?: string; token_id?: string };

    await use(key.key);

    const deleteResponse = await page.request.post(`${rootPath()}/key/delete`, {
      headers: adminHeaders,
      data: { keys: [key.token ?? key.token_id ?? key.key] },
    });
    expect(deleteResponse.status(), await deleteResponse.text()).toBe(200);
  },
  submittedServerName: async ({ submissionRules, teamKey, page }, use) => {
    void submissionRules;
    const name = `e2e_rules_${uniqueSuffix().replaceAll("-", "")}`;
    const registration = await page.request.post(`${rootPath()}/v1/mcp/server/register`, {
      headers: {
        "x-litellm-api-key": teamKey,
        "Content-Type": "application/json",
      },
      data: {
        server_name: name,
        alias: name,
        description: "Submitted from the MCP submission rules E2E test",
        url: localServerUrl,
        transport: "http",
      },
      timeout: 60_000,
    });
    expect(registration.status(), await registration.text()).toBe(201);

    await use(name);

    await deleteMcpServerByName(page, name);
  },
});

test.use({ storageState: ADMIN_STORAGE_PATH });
test.describe.configure({ mode: "serial" });
const rejectionTest = test.extend({ requiredFields: [] });

test("saves and clears the MCP client identity header", async ({ page, networkHeader }) => {
  const body = { field_name: "mcp_client_id_header", config_type: "general_settings" };
  const value = `x-mcp-e2e-${uniqueSuffix()}`;

  await navigateToPage(page, Page.McpServers);
  await page.getByRole("tab", { name: "Network Settings", exact: true }).click();
  const panel = page.getByRole("tabpanel", { name: "Network Settings", exact: true });
  const input = panel.getByRole("textbox", { name: "Client identity header", exact: true });
  const save = panel.getByRole("button", { name: "Save", exact: true });
  const saveAndCapture = async (operation: "update" | "delete", expected: object) => {
    const [response] = await Promise.all([
      page.waitForResponse((res) => res.request().method() === "POST" &&
        new URL(res.url()).pathname === `${rootPath()}/config/field/${operation}`),
      save.click(),
    ]);
    expect(response.request().postDataJSON()).toEqual(expected);
    expect(response.ok()).toBe(true);
  };

  await expect(input).toHaveValue(networkHeader);
  await input.fill(` ${value} `);
  await saveAndCapture("update", { ...body, field_value: value });
  await expect(page.getByText("MCP network settings saved", { exact: true }).first()).toBeVisible();
  await input.fill("");
  await saveAndCapture("delete", body);
});

test("Submission Rules panel shows saved rules, preloads the editor, and save keeps existing rules", async ({ page, adminHeaders, submissionRules, submittedServerName }) => {
  await navigateToPage(page, Page.McpServers);
  await page.getByRole("tab", { name: "Submitted MCPs" }).click();

  await expect(page.getByText("(3 required fields)", { exact: true })).toBeVisible();
  await expect(page.getByText("no rules set", { exact: true })).toHaveCount(0);

  await page.getByText("Submission Rules", { exact: true }).click();
  await expect(page.getByRole("checkbox", { name: /^Description\b/ })).toBeChecked();
  await expect(page.getByRole("checkbox", { name: /^Server URL\b/ })).toBeChecked();
  await expect(page.getByRole("checkbox", { name: /^Alias\b/ })).toBeChecked();
  await expect(page.getByRole("checkbox", { name: /^Auth configured\b/ })).not.toBeChecked();

  await page.getByRole("checkbox", { name: /^Auth configured\b/ }).check();
  await page.getByRole("button", { name: "Save Rules" }).click();
  await expect(page.getByText("Submission rules saved", { exact: true })).toBeVisible();

  await waitForSetting(page.request, adminHeaders, {
    field_name: "mcp_required_fields",
    field_value: [...submissionRules, "auth_type"],
    stored_in_db: true,
  });
  await page.reload();
  await page.getByRole("tab", { name: "Submitted MCPs" }).click();
  const submittedCard = page
    .locator("div")
    .filter({ has: page.getByRole("heading", { name: submittedServerName, exact: true }) })
    .filter({ hasText: /\d+ passing, \d+ failing/ })
    .last();
  await expect(submittedCard.getByRole("heading", { name: submittedServerName, exact: true })).toBeVisible();
  await expect(submittedCard.getByText("3 passing, 1 failing", { exact: true })).toBeVisible();
});

rejectionTest("rejects a submission and records the final review notes after cancel and reopen", async ({ page, submittedServerName }) => {
  await navigateToPage(page, Page.McpServers);
  await page.getByRole("tab", { name: "Submitted MCPs", exact: true }).click();
  await page.getByPlaceholder("Search MCP servers...").fill(submittedServerName);
  await expect(page.getByRole("heading", { name: submittedServerName, exact: true })).toBeVisible();
  const reject = page.getByRole("button", { name: "Reject", exact: true });
  const dialog = page.getByRole("heading", { name: "Reject MCP Server", exact: true }).locator("..");
  const input = dialog.getByPlaceholder("Reason for rejection (optional)");
  await reject.click();
  await input.fill("Discard this draft");
  await dialog.getByRole("button", { name: "Cancel", exact: true }).click();
  await reject.click();
  await expect(input).toHaveValue("");

  const review_notes = "  Please document the connector permissions  ";
  await input.fill(review_notes);
  const [response] = await Promise.all([
    page.waitForResponse((res) => res.request().method() === "PUT" &&
      new URL(res.url()).pathname.startsWith(`${rootPath()}/v1/mcp/server/`) &&
      new URL(res.url()).pathname.endsWith("/reject")),
    dialog.getByRole("button", { name: "Reject", exact: true }).click(),
  ]);
  expect(response.request().postDataJSON()).toEqual({ review_notes });
  expect(response.ok()).toBe(true);
  await expect(dialog).toBeHidden();
  await expect(page.getByRole("button", { name: "Re-approve", exact: true })).toBeVisible();
  await expect(page.getByText("Rejection reason: Please document the connector permissions", { exact: true })).toBeVisible();
});
