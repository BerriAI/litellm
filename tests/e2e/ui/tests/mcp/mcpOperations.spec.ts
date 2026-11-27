import { expect, test } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { deleteMcpServerByName } from "../../helpers/mcp";
import { navigateToPage } from "../../helpers/navigation";
import { masterKey, rootPath, uniqueSuffix } from "../../helpers/traffic";

test.use({ storageState: ADMIN_STORAGE_PATH });

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
