import { test as base, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { deleteMcpServerByName } from "../../helpers/mcp";
import { navigateToPage } from "../../helpers/navigation";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";
import { Page } from "../../fixtures/pages";

const localServerUrl = process.env.MOCK_LLM_URL ?? "http://127.0.0.1:8090/v1";

interface Fixtures {
  adminHeaders: { Authorization: string; "Content-Type": string };
  submissionRules: string[];
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
  submissionRules: async ({ adminHeaders, page }, use) => {
    const settingsResponse = await page.request.get(`${rootPath()}/config/list?config_type=general_settings`, {
      headers: adminHeaders,
    });
    expect(settingsResponse.status(), await settingsResponse.text()).toBe(200);
    const settings = (await settingsResponse.json()) as { field_name?: string; field_value?: unknown; stored_in_db?: boolean }[];
    const row = settings.find((item) => item.field_name === "mcp_required_fields");
    const previousRules = row?.stored_in_db === true ? (row.field_value ?? null) : null;
    const rules = ["description", "url", "alias"];
    const rulesResponse = await page.request.post(`${rootPath()}/config/field/update`, {
      headers: adminHeaders,
      data: { field_name: "mcp_required_fields", field_value: rules, config_type: "general_settings" },
    });
    expect(rulesResponse.status(), await rulesResponse.text()).toBe(200);

    await use(rules);

    const restoreResponse = Array.isArray(previousRules)
      ? await page.request.post(`${rootPath()}/config/field/update`, {
          headers: adminHeaders,
          data: {
            field_name: "mcp_required_fields",
            field_value: previousRules,
            config_type: "general_settings",
          },
        })
      : await page.request.post(`${rootPath()}/config/field/delete`, {
          headers: adminHeaders,
          data: { field_name: "mcp_required_fields", config_type: "general_settings" },
        });
    expect(restoreResponse.status(), await restoreResponse.text()).toBe(200);
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
    });
    expect(registration.status(), await registration.text()).toBe(201);

    await use(name);

    await deleteMcpServerByName(page, name);
  },
});

test.use({ storageState: ADMIN_STORAGE_PATH });

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

  await expect
    .poll(
      async () => {
        const response = await page.request.get(
          `${rootPath()}/config/list?config_type=general_settings`,
          { headers: adminHeaders },
        );
        if (!response.ok()) return null;
        const settings = (await response.json()) as { field_name?: string; field_value?: unknown }[];
        return settings.find((row) => row.field_name === "mcp_required_fields")?.field_value ?? null;
      },
      { timeout: 30_000 },
    )
    .toEqual([...submissionRules, "auth_type"]);

  await page.reload();
  await page.getByRole("tab", { name: "Submitted MCPs" }).click();
  await expect(page.getByText(submittedServerName, { exact: true })).toBeVisible();
  await expect(page.getByText("3 passing, 1 failing", { exact: true })).toBeVisible();
});
