import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import {
  approveMcpSubmissionViaUi,
  createToolsetViaUi,
  registerMcpServerViaApi,
  saveNetworkSettings,
  saveSemanticFilterSettings,
  seedMcpServerViaApi,
} from "../../helpers/mcp";

test.describe("MCP Servers", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Add a custom MCP server via the discovery → custom form", async ({ page }) => {
    await navigateToPage(page, Page.McpServers);

    await page.getByRole("button", { name: /Add New MCP Server/i }).click();
    const discovery = page.locator(".ant-modal:visible").filter({ hasText: "Add MCP Server" });
    await expect(discovery).toBeVisible({ timeout: 5_000 });
    await discovery.getByRole("button", { name: /Custom Server/i }).click();

    const formModal = page.locator(".ant-modal:visible").filter({ hasText: "MCP Server Name" });
    await expect(formModal).toBeVisible({ timeout: 5_000 });

    const uniqueName = `e2e_mcp_${Date.now()}`;
    await formModal.locator('input[id="server_name"]').fill(uniqueName);

    const transportField = formModal.locator(".ant-form-item", { hasText: "Transport Type" });
    await transportField.locator(".ant-select").click();
    await page.locator(".ant-select-dropdown:visible").getByText("Streamable HTTP").click();

    await formModal.locator('input[id="url"]').fill("https://e2e-fake-mcp.test.local/mcp");

    const authSection = formModal.locator(".ant-collapse-item", { hasText: /^Authentication/ });
    const authField = authSection.locator(".ant-form-item").first();
    await authField.locator(".ant-select").click();
    await page.locator(".ant-select-dropdown:visible").getByText("None", { exact: true }).click();

    await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();

    await expect(page.getByText("MCP Server created successfully").first()).toBeVisible({ timeout: 15_000 });
    await expect(page.locator("table tbody").getByText(uniqueName).first()).toBeVisible({ timeout: 10_000 });
  });

  test("Proxy admin can create a toolset", async ({ page, request }) => {
    await seedMcpServerViaApi(request, `e2e_mcp_toolset_seed_${Date.now()}`);
    const toolsetName = `e2e_toolset_${Date.now()}`;
    await createToolsetViaUi(page, toolsetName);
  });

  test("Proxy admin can save semantic filter settings", async ({ page }) => {
    await saveSemanticFilterSettings(page);
  });

  test("Proxy admin can save network settings", async ({ page }) => {
    await saveNetworkSettings(page);
  });

  test("Proxy admin can approve a submitted MCP server", async ({ page, request }) => {
    const serverName = `e2e_mcp_pending_${Date.now()}`;
    await registerMcpServerViaApi(request, serverName);
    await approveMcpSubmissionViaUi(page, serverName);
    // Success toast and the new row in the table. Scope the row lookup to
    // the MCP servers table so the form modal's `server_name` input — which
    // still holds the timestamped value during its close animation — can't
    // satisfy the assertion before the server actually lands in the list.
    await expect(page.getByText("MCP Server created successfully").first()).toBeVisible({ timeout: 15_000 });
    await expect(page.locator("table tbody").getByText(uniqueName).first()).toBeVisible({ timeout: 10_000 });
  });
});
