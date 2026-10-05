import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { deleteMcpServerByName } from "../../helpers/mcp";
import { navigateToPage } from "../../helpers/navigation";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";
import { Page } from "../../fixtures/pages";

test.use({ storageState: ADMIN_STORAGE_PATH });

const localServerUrl = process.env.MOCK_LLM_URL ?? "http://127.0.0.1:8090/v1";

let submittedServerName = "";
let submittedKeyToken = "";
let rulesSnapshotTaken = false;
let previousRules: unknown = null;

test.afterEach(async ({ page }) => {
  if (rulesSnapshotTaken) {
    try {
      const headers = { Authorization: `Bearer ${masterKey()}`, "Content-Type": "application/json" };
      if (Array.isArray(previousRules)) {
        await page.request.post(`${rootPath()}/config/field/update`, {
          headers,
          data: { field_name: "mcp_required_fields", field_value: previousRules, config_type: "general_settings" },
        });
      } else {
        await page.request.post(`${rootPath()}/config/field/delete`, {
          headers,
          data: { field_name: "mcp_required_fields", config_type: "general_settings" },
        });
      }
    } catch {}
  }

  if (submittedServerName) {
    try {
      await deleteMcpServerByName(page, submittedServerName);
    } catch {}
  }

  if (submittedKeyToken) {
    try {
      await page.request.post(`${rootPath()}/key/delete`, {
        headers: {
          Authorization: `Bearer ${masterKey()}`,
          "Content-Type": "application/json",
        },
        data: { keys: [submittedKeyToken] },
      });
    } catch {}
  }

  rulesSnapshotTaken = false;
  previousRules = null;
  submittedServerName = "";
  submittedKeyToken = "";
});

test("Submission Rules panel shows saved rules, preloads the editor, and save keeps existing rules", async ({ page }) => {
  const adminHeaders = {
    Authorization: `Bearer ${masterKey()}`,
    "Content-Type": "application/json",
  };
  const settingsResponse = await page.request.get(`${rootPath()}/config/list?config_type=general_settings`, {
    headers: adminHeaders,
  });
  expect(settingsResponse.status(), await settingsResponse.text()).toBe(200);
  const settings = (await settingsResponse.json()) as { field_name?: string; field_value?: unknown; stored_in_db?: boolean }[];
  const rulesRow = settings.find((row) => row.field_name === "mcp_required_fields");
  previousRules = rulesRow?.stored_in_db === true ? (rulesRow.field_value ?? null) : null;
  rulesSnapshotTaken = true;

  const rulesResponse = await page.request.post(`${rootPath()}/config/field/update`, {
    headers: adminHeaders,
    data: {
      field_name: "mcp_required_fields",
      field_value: ["description", "url", "alias"],
      config_type: "general_settings",
    },
  });
  expect(rulesResponse.status(), await rulesResponse.text()).toBe(200);

  const keyResponse = await page.request.post(`${rootPath()}/key/generate`, {
    headers: adminHeaders,
    data: { team_id: "e2e-team-crud", user_id: "e2e-internal-user" },
  });
  expect(keyResponse.status(), await keyResponse.text()).toBe(200);
  const key = (await keyResponse.json()) as { key: string; token?: string; token_id?: string };
  const teamKey = key.key;
  submittedKeyToken = key.token ?? key.token_id ?? teamKey;

  submittedServerName = `e2e_rules_${uniqueSuffix().replaceAll("-", "")}`;
  const registration = await page.request.post(`${rootPath()}/v1/mcp/server/register`, {
    headers: {
      "x-litellm-api-key": teamKey,
      "Content-Type": "application/json",
    },
    data: {
      server_name: submittedServerName,
      alias: submittedServerName,
      description: "Submitted from the MCP submission rules E2E test",
      url: localServerUrl,
      transport: "http",
    },
  });
  expect(registration.status(), await registration.text()).toBe(201);

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
    .toEqual(["description", "url", "alias", "auth_type"]);

  await page.reload();
  await page.getByRole("tab", { name: "Submitted MCPs" }).click();
  await expect(page.getByText(submittedServerName, { exact: true })).toBeVisible();
  await expect(page.getByText("3 passing, 1 failing", { exact: true })).toBeVisible();
});
