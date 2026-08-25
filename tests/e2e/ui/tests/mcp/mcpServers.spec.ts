import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { deleteMcpServerByName } from "../../helpers/mcp";

// Coverage scope: only the happy-path Streamable HTTP + None auth create flow.
// See E2E_COVERAGE.md (#29 row) for the full list of uncovered MCP surfaces
// — SSE / stdio / OpenAPI transports, API Key / Bearer / OAuth2 / Basic / Token
// / AWS SigV4 auth, edit/delete, BYOK credentials, tool list/call (needs a real
// or mocked MCP server in the e2e fixture stack), and access-group permissions.
test.describe("MCP Servers", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  let createdServerName = "";

  // The server this test creates is unreachable, and the MCP page contacts
  // every server it lists, so leaving it behind slows down every later MCP
  // test. See deleteMcpServerByName for what that actually cost.
  test.afterEach(async ({ page }) => {
    if (createdServerName) await deleteMcpServerByName(page, createdServerName);
  });

  test("Add a custom MCP server via the discovery → custom form", async ({ page }) => {
    await navigateToPage(page, Page.McpServers);

    // Open the discovery modal, then drop into the custom-server form
    await page.getByRole("button", { name: /Add New MCP Server/i }).click();
    const discovery = page.getByRole("dialog").filter({ hasText: "Add MCP Server" });
    await expect(discovery).toBeVisible({ timeout: 5_000 });
    await discovery.getByRole("button", { name: /Custom Server/i }).click();

    const formModal = page.getByRole("dialog").filter({ hasText: "MCP Server Name" });
    await expect(formModal).toBeVisible({ timeout: 5_000 });

    // Name — no spaces or hyphens per validateMCPServerName
    const uniqueName = `e2e_mcp_${Date.now()}`;
    createdServerName = uniqueName;
    await formModal.getByLabel("MCP Server Name").fill(uniqueName);

    // Transport: Streamable HTTP — the only value the proxy actually accepts is "http".
    // Select popups are portaled to the body, so the option lookup is page-scoped.
    await formModal.getByRole("combobox", { name: "Transport Type" }).click();
    await page.getByRole("option", { name: "Streamable HTTP" }).click();

    // URL — use a fake URL; the form just persists it, it doesn't have to be reachable
    await formModal.getByLabel("MCP Server URL").fill("https://e2e-fake-mcp.test.local/mcp");

    // Authentication: None. "Authentication" is exact so it can't also match the
    // "Authentication Value" field that some auth types reveal below it.
    await formModal.getByRole("combobox", { name: "Authentication", exact: true }).click();
    await page.getByRole("option", { name: "None", exact: true }).click();

    // Submit
    await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();

    // Success toast and the new card in the server grid. Scope the lookup to
    // the MCP servers grid so the form modal's `server_name` input — which
    // still holds the timestamped value during its close animation — can't
    // satisfy the assertion before the server actually lands in the list.
    await expect(page.getByText("MCP Server created successfully").first()).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("mcp-servers-grid").getByText(uniqueName).first()).toBeVisible({ timeout: 10_000 });
  });
});
