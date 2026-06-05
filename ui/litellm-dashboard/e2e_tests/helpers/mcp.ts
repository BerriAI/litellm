import { APIRequestContext, expect, Locator, Page as PlaywrightPage } from "@playwright/test";
import { E2E_INTERNAL_USER_API_KEY } from "../constants";
import { Page } from "../fixtures/pages";
import { clickTeamId, dismissFeedbackPopup, navigateToPage } from "./navigation";

export const E2E_MCP_FAKE_URL = "https://e2e-fake-mcp.test.local/mcp";

function masterKey(): string {
  return process.env.LITELLM_MASTER_KEY || "sk-1234";
}

export async function navigateToMcpTab(page: PlaywrightPage, tabName: string | RegExp): Promise<void> {
  await navigateToPage(page, Page.McpServers);
  await page.getByRole("tab", { name: tabName }).click();
}

export async function seedMcpServerViaApi(
  request: APIRequestContext,
  serverName: string,
  options: { allow_all_keys?: boolean } = {},
): Promise<{ server_id: string }> {
  const response = await request.post("/v1/mcp/server", {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: {
      server_name: serverName,
      alias: serverName,
      url: E2E_MCP_FAKE_URL,
      transport: "http",
      auth_type: "none",
      ...options,
    },
  });
  expect(response.ok(), `seed MCP server failed: ${response.status()} ${await response.text()}`).toBeTruthy();
  return response.json();
}

export async function registerMcpServerViaApi(
  request: APIRequestContext,
  serverName: string,
  apiKey = E2E_INTERNAL_USER_API_KEY,
): Promise<{ server_id: string }> {
  const response = await request.post("/v1/mcp/server/register", {
    headers: { Authorization: `Bearer ${apiKey}` },
    data: {
      server_name: serverName,
      alias: serverName,
      url: E2E_MCP_FAKE_URL,
      transport: "http",
      auth_type: "none",
    },
  });
  expect(response.ok(), `register MCP server failed: ${response.status()} ${await response.text()}`).toBeTruthy();
  return response.json();
}

export async function updateTeamMcpPermissionsViaApi(
  request: APIRequestContext,
  teamId: string,
  serverIds: string[],
): Promise<void> {
  const response = await request.post("/team/update", {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: {
      team_id: teamId,
      object_permission: { mcp_servers: serverIds },
    },
  });
  expect(response.ok(), `team MCP update failed: ${response.status()} ${await response.text()}`).toBeTruthy();
}

export async function createCustomMcpServer(
  page: PlaywrightPage,
  serverName = `e2e_mcp_rbac_${Date.now()}`,
): Promise<string> {
  await navigateToPage(page, Page.McpServers);

  await page.getByRole("button", { name: /Add New MCP Server/i }).click();
  const discovery = page.locator(".ant-modal:visible").filter({ hasText: "Add MCP Server" });
  await expect(discovery).toBeVisible({ timeout: 5_000 });
  await discovery.getByRole("button", { name: /Custom Server/i }).click();

  const formModal = page.locator(".ant-modal:visible").filter({ hasText: "MCP Server Name" });
  await expect(formModal).toBeVisible({ timeout: 5_000 });

  await formModal.locator('input[id="server_name"]').fill(serverName);

  const transportField = formModal.locator(".ant-form-item", { hasText: "Transport Type" });
  await transportField.locator(".ant-select").click();
  await page.locator(".ant-select-dropdown:visible").getByText("Streamable HTTP").click();

  await formModal.locator('input[id="url"]').fill(E2E_MCP_FAKE_URL);

  const authSection = formModal.locator(".ant-collapse-item", { hasText: /^Authentication/ });
  const authField = authSection.locator(".ant-form-item").first();
  await authField.locator(".ant-select").click();
  await page.locator(".ant-select-dropdown:visible").getByText("None", { exact: true }).click();

  await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();

  await expect(page.getByText("MCP Server created successfully").first()).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("table tbody").getByText(serverName).first()).toBeVisible({ timeout: 10_000 });

  return serverName;
}

export async function submitCustomMcpServerViaUi(
  page: PlaywrightPage,
  serverName = `e2e_mcp_submit_${Date.now()}`,
): Promise<string> {
  await navigateToPage(page, Page.McpServers);
  await page.getByRole("button", { name: /Submit MCP Server/i }).click();

  const formModal = page.locator(".ant-modal:visible").filter({ hasText: "Submit MCP Server for Review" });
  await expect(formModal).toBeVisible({ timeout: 5_000 });

  await formModal.locator('input[id="server_name"]').fill(serverName);

  const transportField = formModal.locator(".ant-form-item", { hasText: "Transport Type" });
  await transportField.locator(".ant-select").click();
  await page.locator(".ant-select-dropdown:visible").getByText("Streamable HTTP").click();

  await formModal.locator('input[id="url"]').fill(E2E_MCP_FAKE_URL);

  const authSection = formModal.locator(".ant-collapse-item", { hasText: /^Authentication/ });
  const authField = authSection.locator(".ant-form-item").first();
  await authField.locator(".ant-select").click();
  await page.locator(".ant-select-dropdown:visible").getByText("None", { exact: true }).click();

  await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();
  await expect(page.getByText("MCP Server submitted for admin review").first()).toBeVisible({ timeout: 15_000 });

  return serverName;
}

export async function createToolsetViaUi(page: PlaywrightPage, toolsetName: string): Promise<void> {
  await navigateToMcpTab(page, "Toolsets");
  await expect(page.getByText("MCP Toolsets").first()).toBeVisible({ timeout: 10_000 });

  await page.getByRole("button", { name: "New Toolset" }).click();
  const modal = page.locator(".ant-modal:visible").filter({ hasText: "New Toolset" });
  await expect(modal).toBeVisible({ timeout: 5_000 });

  await modal.getByPlaceholder("e.g. github-linear-tools").fill(toolsetName);
  await modal.getByRole("button", { name: "Create Toolset" }).click();

  await expect(page.locator("table tbody").getByText(toolsetName).first()).toBeVisible({ timeout: 15_000 });
}

export async function saveSemanticFilterSettings(page: PlaywrightPage): Promise<void> {
  await navigateToMcpTab(page, "Semantic Filter");
  await expect(page.getByText("Semantic Tool Filtering").first()).toBeVisible({ timeout: 10_000 });

  const topK = page.getByRole("spinbutton", { name: "Top K Results" });
  await expect(topK).toBeVisible({ timeout: 10_000 });
  await topK.fill("11");

  const saveButton = page.getByRole("button", { name: "Save Settings" });
  await expect(saveButton).toBeEnabled({ timeout: 5_000 });
  await saveButton.click();

  await expect(
    page.getByText(/Settings saved successfully|Settings updated successfully/i).first(),
  ).toBeVisible({ timeout: 15_000 });
}

export async function saveNetworkSettings(page: PlaywrightPage, cidr = "10.0.0.0/8"): Promise<void> {
  await navigateToMcpTab(page, "Network Settings");
  await expect(page.getByText("Private IP Ranges").first()).toBeVisible({ timeout: 15_000 });

  const rangeSelect = page.locator(".ant-select").filter({
    has: page.locator(".ant-select-selection-placeholder", { hasText: /Leave empty to use defaults/i }),
  });
  await rangeSelect.click();
  await page.keyboard.type(cidr);
  await page.keyboard.press("Enter");

  await page.getByRole("button", { name: "Save" }).click();
  await expect(page.getByText("Successfully updated value!").first()).toBeVisible({ timeout: 15_000 });
}

export async function approveMcpSubmissionViaUi(page: PlaywrightPage, serverName: string): Promise<void> {
  await navigateToMcpTab(page, /Submitted MCPs/i);
  await expect(page.getByText("Total Submitted").first()).toBeVisible({ timeout: 15_000 });

  const submissionCard = page
    .locator("div.bg-white.border")
    .filter({ has: page.getByRole("heading", { name: serverName, level: 3 }) })
    .first();
  await expect(submissionCard).toBeVisible({ timeout: 10_000 });
  await submissionCard.getByRole("button", { name: "Approve" }).click();

  const dialog = page.locator("div.fixed.inset-0").filter({ hasText: "Approve MCP Server" });
  await expect(dialog).toBeVisible({ timeout: 5_000 });
  await dialog.getByRole("button", { name: "Approve" }).click();

  await expect(page.getByText(new RegExp(`MCP server "${serverName}" approved`)).first()).toBeVisible({
    timeout: 15_000,
  });
  await expect(submissionCard.getByText("Active").first()).toBeVisible({ timeout: 10_000 });
}

export async function openMcpServerDetail(page: PlaywrightPage, serverName: string): Promise<void> {
  const row = page.locator("table tbody tr", { hasText: serverName }).first();
  await expect(row).toBeVisible({ timeout: 10_000 });
  await row.locator("button").first().click();
  await expect(page.getByRole("button", { name: "Back to All Servers" })).toBeVisible({ timeout: 10_000 });
}

export async function selectMcpServerInForm(page: PlaywrightPage, scope: Locator, serverName: string): Promise<void> {
  const field = scope.locator(".ant-form-item", { hasText: /MCP Servers/i });
  await field.locator(".ant-select").click();
  const option = page.locator(".ant-select-dropdown:visible").getByText(serverName).first();
  await expect(option).toBeAttached({ timeout: 10_000 });
  await option.evaluate((el: HTMLElement) => el.click());
  await page.keyboard.press("Escape");
}

export async function assignMcpServerToTeamSettings(
  page: PlaywrightPage,
  teamId: string,
  serverName: string,
): Promise<void> {
  await navigateToPage(page, Page.Teams);
  await dismissFeedbackPopup(page);
  await clickTeamId(page, teamId);

  await page.getByRole("tab", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Edit Settings" }).click();

  const settingsForm = page.locator("form").filter({ hasText: "MCP Servers / Access Groups" });
  await selectMcpServerInForm(page, settingsForm, serverName);

  await page.getByRole("button", { name: "Save Changes" }).click();
  await expect(page.getByText(/Team settings updated|updated successfully/i).first()).toBeVisible({
    timeout: 15_000,
  });
}

export async function expectMcpServerListedInObjectPermissions(
  page: PlaywrightPage,
  serverName: string,
): Promise<void> {
  const permissions = page.locator("text=Object Permissions").locator("..");
  await expect(permissions.getByText("MCP Servers").first()).toBeVisible({ timeout: 10_000 });
  await expect(permissions.getByText(serverName).first()).toBeVisible({ timeout: 10_000 });
}

export async function expectMcpServerListedInTeamObjectPermissions(
  page: PlaywrightPage,
  serverName: string,
): Promise<void> {
  await page.getByRole("tab", { name: "Overview" }).click();
  const objectPermissions = page.locator("text=Object Permissions").locator("..");
  await expect(objectPermissions.getByText(serverName).first()).toBeVisible({ timeout: 10_000 });
}

export async function expectNoMcpAdminControls(page: PlaywrightPage): Promise<void> {
  await expect(page.getByRole("button", { name: /Add New MCP Server/i })).toHaveCount(0);
  await expect(page.getByRole("tab", { name: /Submitted MCPs/i })).toHaveCount(0);
}

export async function expectMcpPageTabsVisible(page: PlaywrightPage): Promise<void> {
  for (const tab of ["All Servers", "Toolsets", "Connect", "Semantic Filter", "Network Settings"]) {
    await expect(page.getByRole("tab", { name: tab })).toBeVisible({ timeout: 5_000 });
  }
}

export async function expectNoMcpServerSettingsTab(page: PlaywrightPage): Promise<void> {
  await expect(page.getByRole("tab", { name: "Overview" })).toBeVisible({ timeout: 5_000 });
  await expect(page.getByRole("tab", { name: "MCP Tools" })).toBeVisible({ timeout: 5_000 });
  await expect(page.getByRole("tab", { name: "Settings" })).toHaveCount(0);
}

export async function attemptSemanticFilterSaveAsNonAdmin(page: PlaywrightPage): Promise<void> {
  await navigateToMcpTab(page, "Semantic Filter");
  await expect(page.getByText("Semantic Tool Filtering").first()).toBeVisible({ timeout: 10_000 });

  const topK = page.getByRole("spinbutton", { name: "Top K Results" });
  await topK.fill("12");

  const saveButton = page.getByRole("button", { name: "Save Settings" });
  await expect(saveButton).toBeEnabled({ timeout: 5_000 });
  await saveButton.click();

  await expect(page.getByText(/Could not update settings|Only proxy admin/i).first()).toBeVisible({
    timeout: 15_000,
  });
}

export async function attemptNetworkSettingsSaveAsNonAdmin(page: PlaywrightPage): Promise<void> {
  await navigateToMcpTab(page, "Network Settings");
  await expect(page.getByText("Private IP Ranges").first()).toBeVisible({ timeout: 15_000 });

  const rangeSelect = page.locator(".ant-select").filter({
    has: page.locator(".ant-select-selection-placeholder", { hasText: /Leave empty to use defaults/i }),
  });
  await rangeSelect.click();
  await page.keyboard.type("192.168.0.0/16");
  await page.keyboard.press("Enter");
  await page.getByRole("button", { name: "Save" }).click();

  await expect(page.getByText("Successfully updated value!")).toHaveCount(0, { timeout: 5_000 });
}

export type McpAuthUiOption =
  | "None"
  | "API Key"
  | "Bearer Token"
  | "Token"
  | "Basic Auth"
  | "OAuth"
  | "AWS SigV4 (Bedrock AgentCore MCPs)";

export async function openAdminCustomMcpServerForm(page: PlaywrightPage): Promise<Locator> {
  await navigateToPage(page, Page.McpServers);
  await page.getByRole("button", { name: /Add New MCP Server/i }).click();
  const discovery = page.locator(".ant-modal:visible").filter({ hasText: "Add MCP Server" });
  await expect(discovery).toBeVisible({ timeout: 5_000 });
  await discovery.getByRole("button", { name: /Custom Server/i }).click();

  const formModal = page.locator(".ant-modal:visible").filter({ hasText: "MCP Server Name" });
  await expect(formModal).toBeVisible({ timeout: 5_000 });
  return formModal;
}

export async function openSubmitMcpServerForm(page: PlaywrightPage): Promise<Locator> {
  await navigateToPage(page, Page.McpServers);
  await page.getByRole("button", { name: /Submit MCP Server/i }).click();
  const formModal = page.locator(".ant-modal:visible").filter({ hasText: "Submit MCP Server for Review" });
  await expect(formModal).toBeVisible({ timeout: 5_000 });
  return formModal;
}

export async function fillMcpServerBasics(formModal: Locator, page: PlaywrightPage, serverName: string): Promise<void> {
  await formModal.locator('input[id="server_name"]').fill(serverName);

  const transportField = formModal.locator(".ant-form-item", { hasText: "Transport Type" });
  await transportField.locator(".ant-select").click();
  await page.locator(".ant-select-dropdown:visible").getByText("Streamable HTTP").click();

  await formModal.locator('input[id="url"]').fill(E2E_MCP_FAKE_URL);
}

export async function selectMcpAuthTypeInForm(
  formModal: Locator,
  page: PlaywrightPage,
  authOption: McpAuthUiOption,
): Promise<void> {
  const authSection = formModal.locator(".ant-collapse-item", { hasText: /^Authentication/ });
  const authField = authSection.locator(".ant-form-item").first();
  await authField.locator(".ant-select").click();

  const option =
    authOption === "None"
      ? page.locator(".ant-select-dropdown:visible").getByText("None", { exact: true })
      : page.locator(".ant-select-dropdown:visible").getByText(authOption, { exact: authOption !== "OAuth" });
  await option.click();
}

export async function fillMcpAuthValue(formModal: Locator, value: string): Promise<void> {
  await formModal.getByPlaceholder("Enter token or secret").fill(value);
}

export async function selectOAuthFlowType(
  formModal: Locator,
  page: PlaywrightPage,
  flow: "m2m" | "interactive",
): Promise<void> {
  const flowField = formModal.locator(".ant-form-item", { hasText: "OAuth Flow Type" });
  await flowField.locator(".ant-select").click();
  const label = flow === "m2m" ? /Machine-to-Machine/i : /Interactive \(PKCE\)/i;
  await page.locator(".ant-select-dropdown:visible").getByText(label).click();
}

export async function fillOAuthM2MCredentials(formModal: Locator): Promise<void> {
  await formModal.getByPlaceholder(/Enter OAuth client ID/i).fill("e2e-oauth-client-id");
  await formModal.getByPlaceholder(/Enter OAuth client secret/i).fill("e2e-oauth-client-secret");
  await formModal.getByPlaceholder("https://auth.example.com/oauth/token").fill("https://e2e-fake-oauth.test/token");
}

export async function fillOAuthInteractiveOverrides(formModal: Locator): Promise<void> {
  await formModal.getByPlaceholder("https://example.com/oauth/authorize").fill("https://e2e-fake-oauth.test/authorize");
  await formModal.getByPlaceholder("https://example.com/oauth/token").fill("https://e2e-fake-oauth.test/token");
}

export async function fillAwsSigV4Credentials(formModal: Locator): Promise<void> {
  await formModal.getByPlaceholder("us-east-1").fill("us-east-1");
  await formModal.getByPlaceholder("bedrock-agentcore").fill("bedrock-agentcore");
}

export async function configureMcpAuthInForm(
  formModal: Locator,
  page: PlaywrightPage,
  authOption: McpAuthUiOption,
): Promise<void> {
  await selectMcpAuthTypeInForm(formModal, page, authOption);

  switch (authOption) {
    case "None":
      return;
    case "API Key":
      await fillMcpAuthValue(formModal, "e2e-mcp-api-key-value");
      return;
    case "Bearer Token":
      await fillMcpAuthValue(formModal, "e2e-mcp-bearer-token");
      return;
    case "Token":
      await fillMcpAuthValue(formModal, "e2e-mcp-token-value");
      return;
    case "Basic Auth":
      await fillMcpAuthValue(formModal, "dGVzdDpzZWNyZXQ=");
      return;
    case "OAuth":
      await selectOAuthFlowType(formModal, page, "m2m");
      await fillOAuthM2MCredentials(formModal);
      return;
    case "AWS SigV4 (Bedrock AgentCore MCPs)":
      await fillAwsSigV4Credentials(formModal);
      return;
  }
}

export async function configureMcpOAuthInteractiveAuth(formModal: Locator, page: PlaywrightPage): Promise<void> {
  await selectMcpAuthTypeInForm(formModal, page, "OAuth");
  await selectOAuthFlowType(formModal, page, "interactive");
  await fillOAuthInteractiveOverrides(formModal);
}

export async function createAdminMcpServerWithAuth(
  page: PlaywrightPage,
  authOption: McpAuthUiOption,
  serverName = `e2e_mcp_auth_${authOption.replace(/\W+/g, "_").toLowerCase()}_${Date.now()}`,
): Promise<string> {
  const formModal = await openAdminCustomMcpServerForm(page);
  await fillMcpServerBasics(formModal, page, serverName);
  await configureMcpAuthInForm(formModal, page, authOption);
  await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();

  await expect(page.getByText("MCP Server created successfully").first()).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("table tbody").getByText(serverName).first()).toBeVisible({ timeout: 10_000 });
  return serverName;
}

export async function createAdminMcpServerWithOAuthInteractiveAuth(
  page: PlaywrightPage,
  serverName = `e2e_mcp_auth_oauth_interactive_${Date.now()}`,
): Promise<string> {
  const formModal = await openAdminCustomMcpServerForm(page);
  await fillMcpServerBasics(formModal, page, serverName);
  await configureMcpOAuthInteractiveAuth(formModal, page);
  await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();

  await expect(page.getByText("MCP Server created successfully").first()).toBeVisible({ timeout: 15_000 });
  await expect(page.locator("table tbody").getByText(serverName).first()).toBeVisible({ timeout: 10_000 });
  return serverName;
}

export async function submitMcpServerWithAuth(
  page: PlaywrightPage,
  authOption: McpAuthUiOption,
  serverName = `e2e_mcp_submit_auth_${Date.now()}`,
): Promise<string> {
  const formModal = await openSubmitMcpServerForm(page);
  await fillMcpServerBasics(formModal, page, serverName);
  await configureMcpAuthInForm(formModal, page, authOption);
  await formModal.getByRole("button", { name: /^Add MCP Server$/ }).click();

  await expect(page.getByText("MCP Server submitted for admin review").first()).toBeVisible({ timeout: 15_000 });
  return serverName;
}

export async function expectMcpServerAuthBadgeOnDetail(
  page: PlaywrightPage,
  serverName: string,
  expectedAuthType: string,
): Promise<void> {
  await openMcpServerDetail(page, serverName);
  await expect(page.getByText("Authentication").first()).toBeVisible({ timeout: 10_000 });
  await expect(page.getByText(expectedAuthType, { exact: true }).first()).toBeVisible({ timeout: 10_000 });
}
