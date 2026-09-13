import { test, expect, Locator } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { createMcpServer, deleteMcpServerByName, openMcpToolsTab } from "../../helpers/mcp";

// Listing and calling MCP tools, which needs a server that really answers; the create-only spec
// points at an unreachable URL on purpose.
//
// This spec makes a read-only network call to DeepWiki's public MCP server, from the proxy rather
// than the browser. It needs no credentials, so there is no secret to leak from a public repo.
//
// A DeepWiki outage turns this red for something that is not a litellm regression. That is left
// visible rather than auto-skipped: skipping on connection trouble also skips when the proxy's own
// MCP client breaks, which is the regression this exists to catch. E2E_SKIP_EXTERNAL_MCP=1 opts out.
const MCP_SERVER_URL = "https://mcp.deepwiki.com/mcp";
const TOOL_NAME = "read_wiki_structure";
const TOOL_ARG_REPO = "BerriAI/litellm";

// Match the h4 heading, not page text: a tool whose description names another tool trips strict mode.
const toolCard = (list: Locator, name: string): Locator =>
  list.locator("h4.font-mono").filter({ hasText: new RegExp(`^${name}$`) });

test.describe("MCP Tools", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });
  test.skip(!!process.env.E2E_SKIP_EXTERNAL_MCP, "E2E_SKIP_EXTERNAL_MCP is set");

  let serverName: string;

  test.beforeEach(async ({ page }) => {
    serverName = await createMcpServer(page, MCP_SERVER_URL);
    await openMcpToolsTab(page, serverName);
  });

  // The MCP page contacts every server it lists, so leaks slow later tests run by run.
  test.afterEach(async ({ page }) => {
    await deleteMcpServerByName(page, serverName);
  });

  test("MCP Tools tab lists the tools the upstream server advertises", async ({ page }) => {
    // Fetched through the proxy on mount, so allow for a cold upstream connection.
    const toolList = page.locator(".mcp-tools-scrollable");
    await expect(toolList).toBeVisible({ timeout: 30_000 });

    // Non-empty would still pass if the proxy returned some other server's tools.
    await expect(toolCard(toolList, TOOL_NAME)).toBeVisible();
    await expect(toolCard(toolList, "ask_question")).toBeVisible();
    await expect(toolCard(toolList, "read_wiki_contents")).toBeVisible();

    // No other tool's name or description contains this string, so exactly one card survives.
    await page.getByPlaceholder("Search tools...").fill(TOOL_NAME);
    await expect(toolList.locator("h4.font-mono")).toHaveCount(1);
    await expect(toolCard(toolList, TOOL_NAME)).toBeVisible();
  });

  test("Calling a tool from the Test Tool panel returns the upstream result", async ({ page }) => {
    const toolList = page.locator(".mcp-tools-scrollable");
    await expect(toolList).toBeVisible({ timeout: 30_000 });

    await toolCard(toolList, TOOL_NAME).click();

    // Selecting a tool swaps the right-hand pane in for the empty state.
    await expect(page.getByText("Test Tool:", { exact: true })).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("Ready to Call Tool")).toBeVisible();

    // The form is generated from the tool's inputSchema, so `repoName` proves the schema
    // round-tripped through the proxy instead of the panel falling back to a generic field.
    const repoInput = page.getByLabel(/repoName/);
    await expect(repoInput).toBeVisible();
    await repoInput.fill(TOOL_ARG_REPO);

    await page.getByRole("button", { name: "Call Tool", exact: true }).click();

    await expect(page.getByText("Tool executed successfully")).toBeVisible({ timeout: 60_000 });
    // read_wiki_structure answers with the repo's outline, so the pane must name the repo.
    await expect(page.getByText(TOOL_ARG_REPO).first()).toBeVisible();

    // A second call is offered rather than the button resetting to its
    // first-run label.
    await expect(page.getByRole("button", { name: "Call Again", exact: true })).toBeVisible();
  });
});
