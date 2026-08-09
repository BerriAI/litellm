import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { createMcpServer, deleteMcpServerByName } from "../../helpers/mcp";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";

/**
 * Editing and deleting an MCP server, verified against the API rather than the
 * success toast.
 *
 * mcpServers.spec.ts only ever CREATES a server, and creation is the one MCP
 * operation nobody has complained about. The reports are all on the other side:
 * an alias rename that needs three or four saves to take, a tool selection that
 * reports success and leaves the row unchanged, a delete that needs two
 * attempts. Every one of those produces a success toast on the failing attempt,
 * so a toast-only assertion cannot tell them apart from working software.
 *
 * The server points at an unreachable URL on purpose. These tests never call
 * the upstream -- they exercise litellm's own persistence -- so depending on a
 * real MCP server would add a network dependency that buys nothing. The specs
 * that do need a live upstream are in mcpTools.spec.ts.
 */
const UNREACHABLE_URL = "https://e2e-fake-mcp.test.local/mcp";

/** GET /v1/mcp/server returns a bare array of servers (useMCPServers types it MCPServer[]). */
async function findServerByName(page: PlaywrightPage, serverName: string): Promise<Record<string, any> | undefined> {
  const servers = await readBack<Record<string, any>[]>(page, "/v1/mcp/server");
  return servers.find((server) => server.server_name === serverName);
}

test.describe("MCP Servers - edit and delete", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  let serverName: string;

  test.beforeEach(async ({ page }) => {
    serverName = await createMcpServer(page, UNREACHABLE_URL);
  });

  // The rename test leaves its server behind, and this spec's servers are the
  // unreachable ones -- exactly the kind that slow the MCP page down for every
  // later test. The delete test's server is already gone; deleting by name is
  // a no-op then.
  test.afterEach(async ({ page }) => {
    await deleteMcpServerByName(page, serverName);
  });

  test("Renaming a server's alias persists", async ({ page }) => {
    const before = await findServerByName(page, serverName);
    expect(before, `created server ${serverName} readable from /v1/mcp/server`).toBeTruthy();

    await page.getByTestId("mcp-servers-grid").getByText(serverName).first().click();
    await expect(page.getByRole("button", { name: /Back to All Servers/i })).toBeVisible({ timeout: 10_000 });

    // exact: the server view also renders a "Network Settings" tab.
    await page.getByRole("tab", { name: "Settings", exact: true }).click();

    // Opening a server from the grid puts the view straight into edit mode
    // (mcp_servers.tsx sets editServer on card click), so the "Edit Settings"
    // button is only rendered when it is NOT already editing. Click it if it is
    // there rather than depending on which of the two states we landed in.
    const editSettings = page.getByRole("button", { name: "Edit Settings" });
    if (await editSettings.isVisible().catch(() => false)) {
      await editSettings.click();
    }

    // Scope to the Settings panel: the create modal stays mounted behind the
    // server view, and it carries its own #alias input and Save button, so an
    // unscoped locator is a strict-mode violation rather than a wrong click.
    const settingsPanel = page.getByRole("tabpanel", { name: "Settings" });

    const newAlias = `${serverName}_renamed`;
    const aliasInput = settingsPanel.locator('input[id="alias"]');
    await expect(aliasInput).toBeVisible({ timeout: 10_000 });
    await aliasInput.fill(newAlias);

    const update = await captureRequestBody(page, { method: "PUT", urlIncludes: "/v1/mcp/server" }, async () => {
      await settingsPanel.getByRole("button", { name: "Save Changes" }).click();
    });
    expect(update.alias, "new alias on the wire").toBe(newAlias);
    // The server being edited must be identified, or the write lands somewhere
    // else entirely -- which is one way a save "succeeds" and changes nothing.
    expect(update.server_id, "update targets the server being edited").toBe(before?.server_id);

    // The reported symptom is a rename that needs repeating, i.e. the first
    // save returns success and does not stick. Only a read-back sees that.
    await expect
      .poll(async () => (await findServerByName(page, serverName))?.alias, {
        message: `alias for ${serverName} did not persist after one save`,
        timeout: 15_000,
      })
      .toBe(newAlias);
  });

  test("Deleting a server removes it", async ({ page }) => {
    expect(await findServerByName(page, serverName), `created server ${serverName} exists`).toBeTruthy();

    const card = page.getByTestId("mcp-servers-grid").locator("div").filter({ hasText: serverName }).first();
    await card.getByRole("button", { name: "Server actions" }).click();
    await page.getByRole("menuitem", { name: "Delete" }).click();

    const dialog = page.getByRole("alertdialog");
    await expect(dialog.getByText("Delete MCP Server?")).toBeVisible({ timeout: 5_000 });
    await dialog.getByRole("button", { name: "Delete", exact: true }).click();

    // One attempt has to be enough. The report is a delete that needs two, and
    // the first attempt is indistinguishable from a working one in the UI.
    await expect
      .poll(async () => await findServerByName(page, serverName), {
        message: `server ${serverName} still present after one delete`,
        timeout: 15_000,
      })
      .toBeUndefined();
  });
});
