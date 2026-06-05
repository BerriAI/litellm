import { test, expect } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  ADMIN_VIEWER_STORAGE_PATH,
  E2E_TEAM_CRUD_ID,
  E2E_UPDATE_LIMITS_KEY_ALIAS,
  E2E_VIEWER_KEY_ALIAS,
  INTERNAL_USER_STORAGE_PATH,
  INTERNAL_VIEWER_STORAGE_PATH,
  TEAM_ADMIN_STORAGE_PATH,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup, clickTeamId } from "../../helpers/navigation";
import {
  assignMcpServerToTeamSettings,
  attemptNetworkSettingsSaveAsNonAdmin,
  attemptSemanticFilterSaveAsNonAdmin,
  createCustomMcpServer,
  expectMcpPageTabsVisible,
  expectMcpServerListedInObjectPermissions,
  expectMcpServerListedInTeamObjectPermissions,
  expectNoMcpAdminControls,
  expectNoMcpServerSettingsTab,
  navigateToMcpTab,
  openMcpServerDetail,
  seedMcpServerViaApi,
  selectMcpServerInForm,
  submitCustomMcpServerViaUi,
  updateTeamMcpPermissionsViaApi,
} from "../../helpers/mcp";

test.describe("MCP Servers RBAC - Proxy Admin", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Proxy admin can create and update an MCP server", async ({ page }) => {
    const serverName = await createCustomMcpServer(page);
    const updatedAlias = `${serverName}_updated`;

    await openMcpServerDetail(page, serverName);

    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();

    const aliasInput = page.locator('input[id="alias"]');
    await expect(aliasInput).toBeVisible({ timeout: 10_000 });
    await aliasInput.fill(updatedAlias);
    await page.getByRole("button", { name: /Save Changes|Update MCP Server/i }).click();

    await expect(page.getByText(/updated successfully|MCP Server updated/i).first()).toBeVisible({ timeout: 15_000 });
    await page.getByRole("button", { name: "Back to All Servers" }).click();
    await expect(page.locator("table tbody").getByText(updatedAlias).first()).toBeVisible({ timeout: 10_000 });
  });

  test("Proxy admin can assign an MCP server to a key on create", async ({ page }) => {
    const serverName = await createCustomMcpServer(page);

    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);
    await page.getByRole("button", { name: /Create New Key/i }).click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    const keyName = `e2e-mcp-key-create-${Date.now()}`;
    await page.getByTestId("base-input").fill(keyName);

    await page.locator(".ant-select-selection-overflow").click();
    await page.locator(".ant-select-dropdown:visible").getByText("All Team Models").click();
    await page.keyboard.press("Escape");

    const modal = page.locator(".ant-modal:visible");
    await modal.getByText("MCP Settings", { exact: true }).click();
    await selectMcpServerInForm(page, modal, serverName);

    await modal.getByRole("button", { name: "Create Key", exact: true }).click();
    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });
    await page.keyboard.press("Escape");

    const keyRow = page.locator("tr", { hasText: keyName });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();
    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await expectMcpServerListedInObjectPermissions(page, serverName);
  });

  test("Proxy admin can assign an MCP server to a key via edit", async ({ page }) => {
    const serverName = await createCustomMcpServer(page);

    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    const keyRow = page.locator("tr", { hasText: E2E_UPDATE_LIMITS_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();
    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();

    const settingsForm = page.locator("form").filter({ hasText: "MCP Servers / Access Groups" });
    await selectMcpServerInForm(page, settingsForm, serverName);

    await page.getByRole("button", { name: "Save Changes" }).click();
    await expect(page.getByText(/updated successfully|Key updated/i).first()).toBeVisible({ timeout: 15_000 });

    await page.getByRole("tab", { name: "Overview" }).click();
    await expectMcpServerListedInObjectPermissions(page, serverName);
  });

  test("Proxy admin can assign MCP servers to a team", async ({ page }) => {
    const serverName = await createCustomMcpServer(page);

    await assignMcpServerToTeamSettings(page, E2E_TEAM_CRUD_ID, serverName);
    await expectMcpServerListedInTeamObjectPermissions(page, serverName);
  });
});

test.describe("MCP Servers RBAC - Internal User", () => {
  test.use({ storageState: INTERNAL_USER_STORAGE_PATH });

  test("Internal user sees Submit MCP Server instead of admin create controls", async ({ page }) => {
    await navigateToPage(page, Page.McpServers);

    await expectNoMcpAdminControls(page);
    await expect(page.getByRole("button", { name: /Submit MCP Server/i })).toBeVisible({ timeout: 5_000 });
  });

  test("Internal user cannot access MCP server Settings tab on server detail", async ({ page, request }) => {
    const serverName = `e2e_mcp_internal_${Date.now()}`;
    await seedMcpServerViaApi(request, serverName);

    await navigateToPage(page, Page.McpServers);
    await openMcpServerDetail(page, serverName);
    await expectNoMcpServerSettingsTab(page);
  });

  test("Internal user only sees MCP servers their team is allowed to access", async ({ page, request }) => {
    const allowedName = `e2e_mcp_allowed_${Date.now()}`;
    const deniedName = `e2e_mcp_denied_${Date.now()}`;

    const allowedServer = await seedMcpServerViaApi(request, allowedName, { allow_all_keys: false });
    await seedMcpServerViaApi(request, deniedName, { allow_all_keys: false });
    await updateTeamMcpPermissionsViaApi(request, E2E_TEAM_CRUD_ID, [allowedServer.server_id]);

    await navigateToPage(page, Page.McpServers);

    await expect(page.locator("table tbody").getByText(allowedName).first()).toBeVisible({ timeout: 10_000 });
    await expect(page.locator("table tbody").getByText(deniedName)).toHaveCount(0);
  });

  test("Internal user cannot edit team MCP settings as a non-admin member", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    await expect(page.getByRole("tab", { name: "Overview" })).toBeVisible({ timeout: 5_000 });
    await expect(page.getByRole("tab", { name: "Settings" })).not.toBeVisible();
  });
});

test.describe("MCP Servers RBAC - Team Admin", () => {
  test.use({ storageState: TEAM_ADMIN_STORAGE_PATH });

  test("Team admin lacks proxy-admin MCP management controls", async ({ page }) => {
    await navigateToPage(page, Page.McpServers);
    await expectNoMcpAdminControls(page);
    await expect(page.getByRole("button", { name: /Submit MCP Server/i })).toBeVisible({ timeout: 5_000 });
    await expectMcpPageTabsVisible(page);
    await navigateToMcpTab(page, "Toolsets");
    await expect(page.getByRole("button", { name: "New Toolset" })).toHaveCount(0);
  });

  test("Team admin cannot access MCP server Settings tab on server detail", async ({ page, request }) => {
    const serverName = `e2e_mcp_team_admin_view_${Date.now()}`;
    await seedMcpServerViaApi(request, serverName);

    await navigateToPage(page, Page.McpServers);
    await openMcpServerDetail(page, serverName);
    await expectNoMcpServerSettingsTab(page);
  });

  test("Team admin can assign MCP servers to their team settings", async ({ page, request }) => {
    const serverName = `e2e_mcp_team_admin_${Date.now()}`;
    await seedMcpServerViaApi(request, serverName);

    await assignMcpServerToTeamSettings(page, E2E_TEAM_CRUD_ID, serverName);
    await expectMcpServerListedInTeamObjectPermissions(page, serverName);
  });
});

test.describe("MCP Page Features RBAC - Internal User", () => {
  test.use({ storageState: INTERNAL_USER_STORAGE_PATH });

  test("Internal user can browse MCP page tabs but not admin-only controls", async ({ page }) => {
    await navigateToPage(page, Page.McpServers);
    await expectMcpPageTabsVisible(page);
    await expectNoMcpAdminControls(page);
    await navigateToMcpTab(page, "Toolsets");
    await expect(page.getByRole("button", { name: "New Toolset" })).toHaveCount(0);
  });

  test("Internal user cannot save semantic filter settings", async ({ page }) => {
    await attemptSemanticFilterSaveAsNonAdmin(page);
  });

  test("Internal user cannot save network settings", async ({ page }) => {
    await attemptNetworkSettingsSaveAsNonAdmin(page);
  });

  test("Internal user can submit an MCP server for review", async ({ page }) => {
    await submitCustomMcpServerViaUi(page, `e2e_mcp_submit_ui_${Date.now()}`);
  });
});

test.describe("MCP Page Features RBAC - Team Admin", () => {
  test.use({ storageState: TEAM_ADMIN_STORAGE_PATH });

  test("Team admin cannot create toolsets or save admin MCP settings", async ({ page }) => {
    await navigateToMcpTab(page, "Toolsets");
    await expect(page.getByRole("button", { name: "New Toolset" })).toHaveCount(0);
    await attemptSemanticFilterSaveAsNonAdmin(page);
    await attemptNetworkSettingsSaveAsNonAdmin(page);
  });
});

test.describe("MCP Page Features RBAC - Admin Viewer", () => {
  test.use({ storageState: ADMIN_VIEWER_STORAGE_PATH });

  test("Admin viewer sees Submitted MCPs tab but cannot create toolsets", async ({ page }) => {
    await navigateToPage(page, Page.McpServers);
    await expect(page.getByRole("tab", { name: /Submitted MCPs/i })).toBeVisible({ timeout: 5_000 });
    await navigateToMcpTab(page, "Toolsets");
    await expect(page.getByRole("button", { name: "New Toolset" })).toHaveCount(0);
  });

  test("Admin viewer cannot save semantic filter or network settings", async ({ page }) => {
    await attemptSemanticFilterSaveAsNonAdmin(page);
    await attemptNetworkSettingsSaveAsNonAdmin(page);
  });
});

test.describe("MCP Page Features RBAC - Internal Viewer", () => {
  test.use({ storageState: INTERNAL_VIEWER_STORAGE_PATH });

  test("Internal viewer can browse MCP tabs but lacks write controls", async ({ page }) => {
    await navigateToPage(page, Page.McpServers);
    await expectMcpPageTabsVisible(page);
    await expectNoMcpAdminControls(page);
    await navigateToMcpTab(page, "Toolsets");
    await expect(page.getByRole("button", { name: "New Toolset" })).toHaveCount(0);
  });

  test("Internal viewer cannot save semantic filter or network settings", async ({ page }) => {
    await attemptSemanticFilterSaveAsNonAdmin(page);
    await attemptNetworkSettingsSaveAsNonAdmin(page);
  });
});

test.describe("MCP Servers RBAC - Internal Viewer", () => {
  test.use({ storageState: INTERNAL_VIEWER_STORAGE_PATH });

  test("Internal viewer can browse MCP servers but lacks admin management controls", async ({ page }) => {
    await navigateToPage(page, Page.McpServers);

    await expect(page.getByText("MCP Servers").first()).toBeVisible({ timeout: 10_000 });
    await expectNoMcpAdminControls(page);
    await expect(page.getByRole("button", { name: /Submit MCP Server/i })).toBeVisible({ timeout: 5_000 });
  });

  test("Internal viewer cannot access MCP server Settings tab on server detail", async ({ page, request }) => {
    const serverName = `e2e_mcp_viewer_${Date.now()}`;
    await seedMcpServerViaApi(request, serverName);

    await navigateToPage(page, Page.McpServers);
    await openMcpServerDetail(page, serverName);
    await expectNoMcpServerSettingsTab(page);
  });

  test("Internal viewer cannot edit key MCP settings", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    const keyRow = page.locator("tr", { hasText: E2E_VIEWER_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();
    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    await expect(page.getByRole("tab", { name: "Settings" })).toHaveCount(0);
  });
});
