import { expect, test } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { deleteMcpServerByName } from "../../helpers/mcp";
import { navigateToPage } from "../../helpers/navigation";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";

test.use({ storageState: ADMIN_STORAGE_PATH });

test("saves and clears the MCP client identity header", async ({ page, request }) => {
  const body = { field_name: "mcp_client_id_header", config_type: "general_settings" };
  const headers = { Authorization: `Bearer ${masterKey()}` };
  const settings = await request.get(`${rootPath()}/config/list?config_type=general_settings`, { headers });
  await expect(settings).toBeOK();
  const rows: { field_name: string; field_value: unknown; stored_in_db: boolean }[] = await settings.json();
  const previous = rows.find((row) => row.field_name === body.field_name);
  const value = `x-mcp-e2e-${uniqueSuffix()}`;

  try {
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

    await input.fill(` ${value} `);
    await saveAndCapture("update", { ...body, field_value: value });
    await expect(page.getByText("MCP network settings saved", { exact: true }).first()).toBeVisible();
    await input.fill("");
    await saveAndCapture("delete", body);
  } finally {
    const restore = previous?.stored_in_db
      ? await request.post(`${rootPath()}/config/field/update`, { headers, data: { ...body, field_value: previous.field_value } })
      : await request.post(`${rootPath()}/config/field/delete`, { headers, data: body });
    await expect(restore).toBeOK();
  }
});

test("blocks invalid connector JSON and shows a successful import in the MCP server list", async ({ page }) => {
  const name = `e2e_mcp_import_${uniqueSuffix().replaceAll("-", "")}`;
  const payload = { mcp_servers: [{ name, url: "http://127.0.0.1:9/mcp", type: "url" }] };
  const writes: unknown[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && new URL(request.url()).pathname === `${rootPath()}/v1/mcp/server/import`) {
      writes.push(request.postDataJSON());
    }
  });

  try {
    await navigateToPage(page, Page.McpServers);
    await page.getByRole("button", { name: "Import from JSON", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Import MCP Connectors", exact: true });
    const json = dialog.getByRole("textbox", { name: "Connector JSON", exact: true });
    const submit = dialog.getByRole("button", { name: "Import", exact: true });
    await json.fill("{");
    await submit.click();
    await expect(dialog.getByRole("alert")).toContainText("Invalid JSON.");
    expect(writes).toEqual([]);

    await json.fill(JSON.stringify(payload));
    await submit.click();
    await expect(dialog.getByText("Imported:", { exact: true }).locator("..")).toContainText(name);
    await dialog.getByRole("button", { name: "Close", exact: true }).last().click();
    await expect(page.getByRole("button", { name: new RegExp(name) })).toBeVisible();
    expect(writes).toEqual([payload]);
  } finally {
    await deleteMcpServerByName(page, name);
  }
});
