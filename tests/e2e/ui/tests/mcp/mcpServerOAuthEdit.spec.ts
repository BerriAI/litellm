import { createHash } from "node:crypto";
import { test, expect, type Locator, type Page } from "@playwright/test";
import { ADMIN_STORAGE_PATH, UI_BASE_URL } from "../../constants";
import { deleteMcpServerByName } from "../../helpers/mcp";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";
import { masterKey } from "../../helpers/traffic";
import { navigateToPage } from "../../helpers/navigation";
import { Page as UiPage } from "../../fixtures/pages";

const MCP_URL = "https://e2e-fake-mcp.test.local/mcp";
const AUTHORIZATION_URL = "https://e2e-oauth.test.local/oauth/authorize";
const TOKEN_URL = "https://e2e-oauth.test.local/oauth/token";
const SCOPES = ["channels:read", "chat:write"];

type McpServerResponse = {
  server_id: string;
  server_name: string;
  credentials?: unknown;
};

type AuthorizeResponse = {
  status: number;
  location: string | null;
};

test.describe("MCP Servers - OAuth2 edit settings", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  let marker = "";
  let serverName = "";
  let serverId = "";
  let clientId = "";
  let clientSecret = "";

  test.beforeEach(async ({ page }, testInfo) => {
    marker = `${testInfo.workerIndex}_${Date.now()}`;
    serverName = `e2e_mcp_oauth_${marker}`;
    clientId = `e2e-client-${marker}`;
    clientSecret = `e2e-secret-${marker}`;

    const response = await page.request.post("/v1/mcp/server", {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: {
        server_name: serverName,
        transport: "http",
        url: MCP_URL,
        auth_type: "oauth2",
        oauth2_flow: "authorization_code",
        authorization_url: AUTHORIZATION_URL,
        token_url: TOKEN_URL,
        credentials: { client_id: clientId, client_secret: clientSecret, scopes: SCOPES },
      },
    });
    expect(response.ok(), `POST /v1/mcp/server failed (${response.status()}): ${await response.text()}`).toBe(true);

    const created = (await response.json()) as McpServerResponse;
    serverId = created.server_id;
    expect(serverId, `server id returned for ${serverName}`).toBeTruthy();
  });

  test.afterEach(async ({ page }, testInfo) => {
    if (testInfo.status !== "passed") {
      const screenshotPath = testInfo.outputPath("failure.png");
      await page.screenshot({ path: screenshotPath, fullPage: true });
      await testInfo.attach("failure-screenshot", { path: screenshotPath, contentType: "image/png" });
    }
    await deleteMcpServerByName(page, serverName);
  });

  async function requestAuthorizeRedirect(page: Page): Promise<AuthorizeResponse> {
    const endpoint = new URL(`/${serverName}/authorize`, UI_BASE_URL);
    const redirectUri = new URL("/ui", UI_BASE_URL).toString();
    endpoint.searchParams.set("redirect_uri", redirectUri);
    endpoint.searchParams.set("state", `state-${marker}`);
    endpoint.searchParams.set("code_challenge", createHash("sha256").update(marker).digest("base64url"));
    endpoint.searchParams.set("code_challenge_method", "S256");
    endpoint.searchParams.set("response_type", "code");

    const response = await page.request.get(endpoint.toString(), { maxRedirects: 0 });
    return { status: response.status(), location: response.headers()["location"] ?? null };
  }

  function assertAuthorizeLocation(location: string): void {
    const upstream = new URL(location, UI_BASE_URL);
    expect(`${upstream.origin}${upstream.pathname}`, "authorize redirect targets the configured OAuth URL").toBe(
      AUTHORIZATION_URL,
    );
    expect(upstream.searchParams.get("client_id"), "authorize redirect uses the stored client_id").toBe(clientId);
    expect((upstream.searchParams.get("scope") ?? "").split(/\s+/), "authorize redirect includes stored scopes").toEqual(
      expect.arrayContaining(SCOPES),
    );
  }

  async function openServerSettings(page: Page): Promise<Locator> {
    await navigateToPage(page, UiPage.McpServers);
    await page.getByTestId("mcp-servers-grid").getByText(serverName).first().click();
    await expect(page.getByRole("button", { name: /Back to All Servers/i })).toBeVisible({ timeout: 10_000 });
    await page.getByRole("tab", { name: "Settings", exact: true }).click();

    const editSettings = page.getByRole("button", { name: "Edit Settings" });
    if (await editSettings.isVisible().catch(() => false)) {
      await editSettings.click();
    }
    return page.getByRole("tabpanel", { name: "Settings" });
  }

  test("Saving an OAuth2 server without edits keeps its client_id and scopes", async ({ page }) => {
    const initial = await requestAuthorizeRedirect(page);
    await test.info().attach("oauth-initial-authorize", {
      body: JSON.stringify({ serverName, clientId, status: initial.status, location: initial.location }, null, 2),
      contentType: "application/json",
    });
    console.info(`[oauth-edit-save] INITIAL_LOCATION=${initial.location ?? "null"}`);
    expect(initial.status, "authorize endpoint redirects before editing").toBeGreaterThanOrEqual(300);
    expect(initial.status, "authorize endpoint redirects before editing").toBeLessThan(400);
    expect(initial.location, "authorize endpoint provides an upstream Location before editing").toBeTruthy();
    assertAuthorizeLocation(initial.location ?? "");

    const settingsPanel = await openServerSettings(page);
    const putBody = await captureRequestBody(
      page,
      { method: "PUT", urlIncludes: "/v1/mcp/server" },
      async () => {
        await settingsPanel.getByRole("button", { name: "Save Changes" }).first().click();
      },
    );
    let postSaveLocation: string | null = null;
    try {
      expect(putBody.server_id, "update targets the created MCP server").toBe(serverId);
      await expect(page.getByText("MCP Server updated successfully").first()).toBeVisible({ timeout: 15_000 });
      await expect
        .poll(
          async () => {
            const response = await requestAuthorizeRedirect(page);
            postSaveLocation = response.location;
            if (!response.location) {
              return { clientId: null, scopes: [] };
            }
            const upstream = new URL(response.location, UI_BASE_URL);
            const scopes = (upstream.searchParams.get("scope") ?? "").split(/\s+/);
            return {
              clientId: upstream.searchParams.get("client_id"),
              scopes: SCOPES.filter((scope) => scopes.includes(scope)),
            };
          },
          {
            message: `authorize redirect for ${serverName} did not retain its client_id and scopes`,
            timeout: 15_000,
          },
        )
        .toEqual({ clientId, scopes: SCOPES });
    } finally {
      await test.info().attach("oauth-edit-save-evidence", {
        body: JSON.stringify(
          {
            serverName,
            clientId,
            initialAuthorizeLocation: initial.location,
            putBody,
            putCredentials: putBody.credentials,
            postSaveAuthorizeLocation: postSaveLocation,
          },
          null,
          2,
        ),
        contentType: "application/json",
      });
    }

    expect(postSaveLocation, "authorize redirect after saving without edits").toBeTruthy();
    console.info(
      `[oauth-edit-save] PUT_BODY=${JSON.stringify(putBody)} POST_SAVE_LOCATION=${postSaveLocation ?? "null"}`,
    );
    assertAuthorizeLocation(postSaveLocation ?? "");
  });

  test("Edit form reopens with the stored scopes", async ({ page }) => {
    const server = await readBack<McpServerResponse>(page, `/v1/mcp/server/${serverId}`);
    await test.info().attach("oauth-edit-form-readback", {
      body: JSON.stringify({ serverName, clientId, credentials: server.credentials }, null, 2),
      contentType: "application/json",
    });
    console.info(`[oauth-edit-form] GET_CREDENTIALS=${JSON.stringify(server.credentials)}`);

    const settingsPanel = await openServerSettings(page);
    for (const scope of SCOPES) {
      await expect(settingsPanel.getByText(scope, { exact: true }), `edit form shows ${scope}`).toBeVisible();
    }

    const clientIdInput = settingsPanel.getByLabel(/Client ID/i).first();
    if (await clientIdInput.count()) {
      await expect(clientIdInput, "edit form shows the configured client_id").toHaveValue(clientId);
    }
  });
});
