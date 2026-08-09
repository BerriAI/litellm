import { test, expect, Locator } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { createMcpServer, deleteMcpServerByName, openMcpToolsTab } from "../../helpers/mcp";

// Covers the two MCP manual-QA items that mcpServers.spec.ts cannot: listing a
// server's tools, and actually calling one. Both need an MCP server that really
// answers, which the create-only spec deliberately avoids (it points at an
// unreachable .test.local URL).
//
// THIS SPEC MAKES A NETWORK CALL TO A THIRD PARTY, and that is a real cost, so
// it is stated plainly rather than buried:
//
//   - The upstream is DeepWiki's public MCP server. It needs no credentials
//     (Streamable HTTP, auth None), which is what makes it usable from a public
//     repo -- there is no secret to leak.
//   - The call is made by the PROXY, not the browser, so the proxy pod is what
//     needs egress. Nothing in the e2e chart restricts that: the only
//     NetworkPolicies the stack renders belong to the bundled postgresql and
//     redis subcharts.
//   - It is read-only (read_wiki_structure returns a repo's doc outline) and
//     answered in well under a second when measured directly.
//
// If DeepWiki has an outage this spec goes red for a reason that is not a
// litellm regression. That failure is left VISIBLE by default rather than
// auto-skipped, because a spec that silently skips on connection trouble also
// silently skips when the proxy's MCP client breaks -- which is the exact
// regression it exists to catch. Set E2E_SKIP_EXTERNAL_MCP=1 to opt out
// explicitly when the upstream is known-bad.
const MCP_SERVER_URL = "https://mcp.deepwiki.com/mcp";
const TOOL_NAME = "read_wiki_structure";
const TOOL_ARG_REPO = "BerriAI/litellm";

// Each tool card renders its name in an `h4.font-mono` and its description in a
// sibling <p>. Matching the heading rather than page text keeps a tool whose
// DESCRIPTION mentions another tool's name from tripping strict mode.
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

  // Servers outlive the test that made them, and the MCP page contacts every
  // one it lists, so leaked servers make later MCP tests slower run by run.
  test.afterEach(async ({ page }) => {
    await deleteMcpServerByName(page, serverName);
  });

  test("MCP Tools tab lists the tools the upstream server advertises", async ({ page }) => {
    // The list is fetched through the proxy on tab mount, so allow for a cold
    // upstream connection rather than the ~250ms a warm direct call takes.
    const toolList = page.locator(".mcp-tools-scrollable");
    await expect(toolList).toBeVisible({ timeout: 30_000 });

    // Assert on tool names DeepWiki actually advertises. Asserting merely that
    // the list is non-empty would still pass if the proxy returned some other
    // server's tools.
    await expect(toolCard(toolList, TOOL_NAME)).toBeVisible();
    await expect(toolCard(toolList, "ask_question")).toBeVisible();
    await expect(toolCard(toolList, "read_wiki_contents")).toBeVisible();

    // Search narrows the list. No other tool's name or description contains
    // this string, so exactly one card must survive.
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

    // The arguments form is generated from the tool's own inputSchema --
    // MCPToolArgumentsForm renders one antd Form.Item per property, named after
    // the property, so `repoName` is proof the schema round-tripped through the
    // proxy rather than the panel falling back to its generic "input" field.
    const repoInput = page.locator('input[id="repoName"]');
    await expect(repoInput).toBeVisible();
    await repoInput.fill(TOOL_ARG_REPO);

    await page.getByRole("button", { name: "Call Tool", exact: true }).click();

    await expect(page.getByText("Tool executed successfully")).toBeVisible({ timeout: 60_000 });
    // Content assertion, not just the success chrome: read_wiki_structure
    // answers with the repo's page outline, so the result pane must contain
    // the repo it was asked about.
    await expect(page.getByText(TOOL_ARG_REPO).first()).toBeVisible();

    // A second call is offered rather than the button resetting to its
    // first-run label.
    await expect(page.getByRole("button", { name: "Call Again", exact: true })).toBeVisible();
  });
});
