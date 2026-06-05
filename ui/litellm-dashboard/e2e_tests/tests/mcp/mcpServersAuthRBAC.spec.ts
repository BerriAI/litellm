import { test } from "@playwright/test";
import { INTERNAL_USER_STORAGE_PATH } from "../../constants";
import { expectMcpServerAuthBadgeOnDetail, submitMcpServerWithAuth } from "../../helpers/mcp";

test.describe("MCP Server Auth RBAC", () => {
  test.use({ storageState: INTERNAL_USER_STORAGE_PATH });

  test("Internal user can submit MCP server with API Key auth for review", async ({ page }) => {
    const serverName = await submitMcpServerWithAuth(page, "API Key");
    await expectMcpServerAuthBadgeOnDetail(page, serverName, "api_key");
  });

  test("Internal user can submit MCP server with Bearer Token auth for review", async ({ page }) => {
    const serverName = await submitMcpServerWithAuth(page, "Bearer Token");
    await expectMcpServerAuthBadgeOnDetail(page, serverName, "bearer_token");
  });
});
