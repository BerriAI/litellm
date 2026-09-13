import { expect, Page as PwPage } from "@playwright/test";
import { navigateToPage } from "./navigation";
import { Page } from "../fixtures/pages";
import { masterKey } from "./traffic";

/** Creates an MCP server through the UI's discovery to custom-form flow and returns its name. */
export async function createMcpServer(page: PwPage, url: string): Promise<string> {
  await navigateToPage(page, Page.McpServers);

  await page.getByRole("button", { name: /Add New MCP Server/i }).click();
  const discovery = page.getByRole("dialog").filter({ hasText: "Add MCP Server" });
  await expect(discovery).toBeVisible({ timeout: 5_000 });
  await discovery.getByRole("button", { name: /Custom Server/i }).click();

  const formModal = page.getByRole("dialog").filter({ hasText: "MCP Server Name" });
  await expect(formModal).toBeVisible({ timeout: 5_000 });

  // validateMCPServerName rejects spaces and hyphens; the worker index avoids a same-millisecond collision.
  const name = `e2e_mcp_${process.env.TEST_WORKER_INDEX ?? "0"}_${Date.now()}`;
  await formModal.getByLabel("MCP Server Name").fill(name);

  // Select popups are portaled to the body, so the option lookup is page-scoped, not modal-scoped.
  await formModal.getByRole("combobox", { name: "Transport Type" }).click();
  await page.getByRole("option", { name: "Streamable HTTP" }).click();

  await formModal.getByLabel("MCP Server URL").fill(url);

  await formModal.getByRole("combobox", { name: "Authentication", exact: true }).click();
  await page.getByRole("option", { name: "None", exact: true }).click();

  await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();
  await expect(page.getByText("MCP Server created successfully").first()).toBeVisible({ timeout: 15_000 });

  const card = page.getByTestId("mcp-servers-grid").getByText(name).first();
  await expect(card).toBeVisible({ timeout: 10_000 });
  return name;
}

/**
 * Deletes every server carrying `serverName`. Leaked servers break unrelated MCP specs: the page
 * reaches out to each one it lists, so unreachable leftovers stall networkidle until it times out.
 * Errors are swallowed because this runs from afterEach.
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
    // best effort, see above
  }
}

/** Opens a server from the grid and switches to its MCP Tools tab. */
export async function openMcpToolsTab(page: PwPage, serverName: string): Promise<void> {
  await page.getByTestId("mcp-servers-grid").getByText(serverName).first().click();
  await expect(page.getByRole("button", { name: /Back to All Servers/i })).toBeVisible({ timeout: 10_000 });
  await page.getByRole("tab", { name: "MCP Tools" }).click();
}
