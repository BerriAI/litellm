import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";

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
    const authField = formModal.locator(".ant-form-item", { hasText: /auth type/i }).first();
    await authField.locator(".ant-select").click();
    await page.locator(".ant-select-dropdown:visible").getByText("None", { exact: true }).click();

    // Submit
    await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();

    // Success toast and the new row in the table
    await expect(page.getByText("MCP Server created successfully").first())
      .toBeVisible({ timeout: 15_000 });
    await expect(page.getByText(uniqueName).first()).toBeVisible({ timeout: 10_000 });
  });
});
