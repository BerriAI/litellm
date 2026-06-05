import { test } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import {
  createAdminMcpServerWithAuth,
  createAdminMcpServerWithOAuthInteractiveAuth,
  expectMcpServerAuthBadgeOnDetail,
  type McpAuthUiOption,
} from "../../helpers/mcp";

const AUTH_CREATE_CASES: Array<{ label: string; option: McpAuthUiOption; badge: string }> = [
  { label: "None", option: "None", badge: "none" },
  { label: "API Key", option: "API Key", badge: "api_key" },
  { label: "Bearer Token", option: "Bearer Token", badge: "bearer_token" },
  { label: "Token", option: "Token", badge: "token" },
  { label: "Basic Auth", option: "Basic Auth", badge: "basic" },
  { label: "OAuth M2M", option: "OAuth", badge: "oauth2" },
  { label: "AWS SigV4", option: "AWS SigV4 (Bedrock AgentCore MCPs)", badge: "aws_sigv4" },
];

test.describe("MCP Server Auth Types", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  for (const authCase of AUTH_CREATE_CASES) {
    test(`Proxy admin can create MCP server with ${authCase.label} auth`, async ({ page }) => {
      const serverName = await createAdminMcpServerWithAuth(page, authCase.option);
      await expectMcpServerAuthBadgeOnDetail(page, serverName, authCase.badge);
    });
  }

  test("Proxy admin can create MCP server with OAuth Interactive (PKCE) auth", async ({ page }) => {
    const serverName = await createAdminMcpServerWithOAuthInteractiveAuth(page);
    await expectMcpServerAuthBadgeOnDetail(page, serverName, "oauth2");
  });
});
