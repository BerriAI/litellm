import { expect, Page as PwPage } from "@playwright/test";
import { navigateToPage } from "./navigation";
import { Page } from "../fixtures/pages";
import { masterKey } from "./traffic";

/**
 * Creates an MCP server through the UI's discovery -> custom form flow and
 * returns the name it was given.
 *
 * Transport is always Streamable HTTP and auth is always None: those are the
 * only combination the rest of this file's callers exercise, and "http" is the
 * only transport value the proxy actually accepts.
 *
 * mcpServers.spec.ts still carries its own inline copy of this flow. It
 * predates this helper and is green, so it was left alone rather than
 * refactored blind; fold it in the next time that spec is touched.
 */
export async function createMcpServer(page: PwPage, url: string): Promise<string> {
  await navigateToPage(page, Page.McpServers);

  await page.getByRole("button", { name: /Add New MCP Server/i }).click();
  const discovery = page.getByRole("dialog").filter({ hasText: "Add MCP Server" });
  await expect(discovery).toBeVisible({ timeout: 5_000 });
  await discovery.getByRole("button", { name: /Custom Server/i }).click();

  const formModal = page.locator(".ant-modal:visible").filter({ hasText: "MCP Server Name" });
  await expect(formModal).toBeVisible({ timeout: 5_000 });

  // validateMCPServerName rejects spaces and hyphens. The worker index is in
  // the name because at workers>1 two workers can reach Date.now() in the same
  // millisecond, and a duplicate name makes the create fail rather than the
  // assertion.
  const name = `e2e_mcp_${process.env.TEST_WORKER_INDEX ?? "0"}_${Date.now()}`;
  await formModal.locator('input[id="server_name"]').fill(name);

  const transportField = formModal.locator(".ant-form-item", { hasText: "Transport Type" });
  await transportField.locator(".ant-select").click();
  await page.locator(".ant-select-dropdown:visible").getByText("Streamable HTTP").click();

  await formModal.locator('input[id="url"]').fill(url);

  // The auth_type Form.Item has no label prop (CreateMCPServer.tsx), so anchor
  // on the enclosing Collapse panel instead of label text.
  const authSection = formModal.locator(".ant-collapse-item", { hasText: /^Authentication/ });
  await authSection.locator(".ant-form-item").first().locator(".ant-select").click();
  await page.locator(".ant-select-dropdown:visible").getByText("None", { exact: true }).click();

  await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();
  await expect(page.getByText("MCP Server created successfully").first()).toBeVisible({ timeout: 15_000 });

  const card = page.getByTestId("mcp-servers-grid").getByText(name).first();
  await expect(card).toBeVisible({ timeout: 10_000 });
  return name;
}

/**
 * Deletes every server carrying `serverName`, through the API.
 *
 * Cleaning up is not housekeeping here, it is what keeps the MCP specs
 * runnable. Servers persist for the life of the database, the MCP page reaches
 * out to each one it lists, and a server whose URL does not answer holds that
 * up. Measured on a local stack: with eleven leaked servers, eight of them
 * pointing at an unreachable host, navigateToPage's networkidle wait stopped
 * settling inside 30s and every MCP spec failed -- including the ones that
 * leaked nothing. Deleting the leftovers made all five pass again.
 *
 * Failures here are swallowed on purpose: this runs from afterEach, and a
 * cleanup error reported as a test failure hides whatever the test itself
 * found.
 */
export async function deleteMcpServerByName(page: PwPage, serverName: string): Promise<void> {
  const headers = { Authorization: `Bearer ${masterKey()}` };
  try {
    const res = await page.request.get("/v1/mcp/server", { headers });
    if (!res.ok()) return;
    const servers = (await res.json()) as { server_id: string; server_name?: string }[];
    for (const server of servers.filter((candidate) => candidate.server_name === serverName)) {
      await page.request.delete(`/v1/mcp/server/${server.server_id}`, { headers });
    }
  } catch {
    // best effort -- see above
  }
}

/** Opens a server from the grid and switches to its MCP Tools tab. */
export async function openMcpToolsTab(page: PwPage, serverName: string): Promise<void> {
  await page.getByTestId("mcp-servers-grid").getByText(serverName).first().click();
  await expect(page.getByRole("button", { name: /Back to All Servers/i })).toBeVisible({ timeout: 10_000 });
  await page.getByRole("tab", { name: "MCP Tools" }).click();
}
