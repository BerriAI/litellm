import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";

// Coverage scope: only the happy-path Streamable HTTP + None auth create flow.
// See E2E_COVERAGE.md (#29 row) for the full list of uncovered MCP surfaces
// — SSE / stdio / OpenAPI transports, API Key / Bearer / OAuth2 / Basic / Token
// / AWS SigV4 auth, edit/delete, BYOK credentials, tool list/call (needs a real
// or mocked MCP server in the e2e fixture stack), and access-group permissions.
test.describe("MCP Servers", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Add a custom MCP server via the discovery → custom form", async ({ page }) => {
    await navigateToPage(page, Page.McpServers);

    // Open the discovery modal, then drop into the custom-server form
    await page.getByRole("button", { name: /Add New MCP Server/i }).click();
    const discovery = page.locator(".ant-modal:visible").filter({ hasText: "Add MCP Server" });
    await expect(discovery).toBeVisible({ timeout: 5_000 });
    await discovery.getByRole("button", { name: /Custom Server/i }).click();

    const formModal = page.locator(".ant-modal:visible").filter({ hasText: "MCP Server Name" });
    await expect(formModal).toBeVisible({ timeout: 5_000 });

    // Name — no spaces or hyphens per validateMCPServerName
    const uniqueName = `e2e_mcp_${Date.now()}`;
    await formModal.locator('input[id="server_name"]').fill(uniqueName);

    // Transport: Streamable HTTP — the only value the proxy actually accepts is "http"
    const transportField = formModal.locator(".ant-form-item", { hasText: "Transport Type" });
    await transportField.locator(".ant-select").click();
    await page.locator(".ant-select-dropdown:visible").getByText("Streamable HTTP").click();

    // URL — use a fake URL; the form just persists it, it doesn't have to be reachable
    await formModal.locator('input[id="url"]').fill("https://e2e-fake-mcp.test.local/mcp");

    // Authentication: None
    // The auth_type Form.Item has no label prop (create_mcp_server.tsx:795), so
    // it can't be anchored by label text. Scope via the enclosing Collapse
    // panel ("Authentication") instead — that anchor is stable even if the
    // placeholder copy changes.
    const authSection = formModal.locator(".ant-collapse-item", { hasText: /^Authentication/ });
    const authField = authSection.locator(".ant-form-item").first();
    await authField.locator(".ant-select").click();
    await page.locator(".ant-select-dropdown:visible").getByText("None", { exact: true }).click();

    // Submit
    await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();

    // No teardown needed — the e2e runner spins up a fresh DB per invocation.

    // Success toast and the new card in the server grid. Scope the lookup to
    // the MCP servers grid so the form modal's `server_name` input — which
    // still holds the timestamped value during its close animation — can't
    // satisfy the assertion before the server actually lands in the list.
    await expect(page.getByText("MCP Server created successfully").first()).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("mcp-servers-grid").getByText(uniqueName).first()).toBeVisible({ timeout: 10_000 });
  });
});
