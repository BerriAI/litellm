/**
 * E2E test: MCP OAuth flow (Interactive / PKCE)
 *
 * Two-layer regression coverage:
 *
 *  Layer 1 — API assertion (catches the original auth bug directly)
 *  ---------------------------------------------------------------
 *  Makes a real GET to /v1/mcp/server/oauth/{id}/authorize with NO
 *  Authorization header. If someone re-adds `user_api_key_auth` to
 *  that route, the proxy returns 401 and this assertion fails immediately.
 *  With auth absent the proxy returns 404 (unknown server) or 307
 *  (valid server), never 401.
 *
 *  Layer 2 — Full UI flow (catches UI / OAuth wiring regressions)
 *  ---------------------------------------------------------------
 *  Logs in, fills the "Add MCP Server" form with OAuth settings, clicks
 *  "Authorize & Fetch Token", and asserts "Token fetched." appears.
 *
 *  Intercept strategy for Layer 2:
 *    A. /v1/mcp/server/oauth/{*}/authorize* — return an HTML page that
 *       writes the fake OAuth result to sessionStorage (same encoding as
 *       setSecureItem) then navigates back, so the real resumeOAuthFlow()
 *       hook picks it up.
 *    B. POST /v1/mcp/server/oauth/{*}/token — return a mock token.
 */
import { test, expect } from "@playwright/test";

const BASE_URL = "http://localhost:4000";
const MOCK_OAUTH_SERVER = "http://localhost:8080";
const FAKE_CODE = "e2e-fake-auth-code";

test.describe("MCP OAuth - Authorize & Fetch Token", () => {
  // =========================================================================
  // Layer 1: direct API check — catches the original "401 auth added" bug
  // =========================================================================
  test("authorize endpoint must be accessible without an API key", async ({ request }) => {
    // Use a nonexistent server ID. Without auth the proxy returns 404 (server
    // not found). With auth re-added it returns 401 before touching the DB.
    const resp = await request.get(
      BASE_URL + "/v1/mcp/server/oauth/regression-check/authorize" +
        "?redirect_uri=http%3A%2F%2Flocalhost%3A4000%2Fui%2Fmcp%2Foauth%2Fcallback" +
        "&state=regression-test" +
        "&response_type=code" +
        "&code_challenge=abc123" +
        "&code_challenge_method=S256" +
        "&client_id=regression-check",
      { failOnStatusCode: false }
    );
    // 401 = auth gate was added. Any other status means no auth gate.
    expect(
      resp.status(),
      "authorize endpoint returned 401 — user_api_key_auth was added back"
    ).not.toBe(401);
  });

  // =========================================================================
  // Layer 2: full UI flow — catches UI / OAuth wiring regressions
  // =========================================================================
  test("creates an OAuth MCP server and completes the authorize flow", async ({ page }) => {
    const serverName = "e2e_mcp_oauth_" + Date.now();

    // ---- login as admin --------------------------------------------------
    await page.goto(BASE_URL + "/ui/login");
    await page.getByPlaceholder("Enter your username").fill("admin");
    await page.getByPlaceholder("Enter your password").fill(
      process.env.LITELLM_MASTER_KEY || "sk-1234"
    );
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await page.waitForURL(
      function(url) { return url.pathname.startsWith("/ui") && !url.pathname.includes("/login"); },
      { timeout: 30000 }
    );
    const dismiss = page.getByText("Don't ask me again");
    if (await dismiss.isVisible({ timeout: 2000 }).catch(function() { return false; })) {
      await dismiss.click();
    }

    // ---- intercept A: proxy's authorize endpoint -------------------------
    // We respond with HTML that writes the fake OAuth result to sessionStorage
    // (using the same encode() logic as setSecureItem) then navigates back to
    // the MCP servers page — so resumeOAuthFlow() picks it up naturally.
    //
    // NOTE: this intercept runs BEFORE the proxy processes the request.
    // Layer 1 (above) covers the auth-on-authorize regression separately.
    await page.route("**/v1/mcp/server/oauth/*/authorize*", async function(route) {
      const url = new URL(route.request().url());
      const clientState = url.searchParams.get("state") || "";

      const encodedPayload = Buffer.from(
        encodeURIComponent(
          JSON.stringify({ type: "litellm-mcp-oauth", code: FAKE_CODE, state: clientState })
        ).replace(/%([0-9A-F]{2})/g, function(_, p1) {
          return String.fromCharCode(parseInt(p1, 16));
        })
      ).toString("base64");

      const html = "<!DOCTYPE html><html><script>" +
        "window.sessionStorage.setItem('litellm-mcp-oauth-result', '" + encodedPayload + "');" +
        "window.location.replace('" + BASE_URL + "/ui?page=mcp-servers');" +
        "</script></html>";

      await route.fulfill({ status: 200, contentType: "text/html", body: html });
    });

    // ---- intercept B: token exchange -------------------------------------
    await page.route("**/v1/mcp/server/oauth/*/token", async function(route) {
      if (route.request().method() !== "POST") {
        await route.continue();
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        headers: { "Cache-Control": "no-store", "Pragma": "no-cache" },
        body: JSON.stringify({
          access_token: "mock-e2e-access-token",
          token_type: "Bearer",
          expires_in: 3600,
        }),
      });
    });

    // ---- navigate to MCP servers page ------------------------------------
    await page.goto(BASE_URL + "/ui?page=mcp-servers");
    await expect(page.getByText("MCP Servers").first()).toBeVisible({ timeout: 20000 });
    const dismiss2 = page.getByText("Don't ask me again");
    if (await dismiss2.isVisible({ timeout: 2000 }).catch(function() { return false; })) {
      await dismiss2.click();
    }

    // ---- open discovery -> custom server -> create modal -----------------
    await page.getByRole("button", { name: "+ Add New MCP Server" }).click();
    await expect(page.getByText("+ Custom Server").first()).toBeVisible({ timeout: 10000 });
    await page.getByText("+ Custom Server").first().click();
    await expect(page.getByRole("heading", { name: "Add New MCP Server" })).toBeVisible({ timeout: 15000 });

    // ---- fill the form ---------------------------------------------------
    await page.getByPlaceholder("e.g., GitHub_MCP, Zapier_MCP, etc.").first().fill(serverName);
    await page.locator(".ant-select", { hasText: "Select transport" }).click();
    await page.locator(".ant-select-dropdown:visible").getByText("Streamable HTTP (Recommended)").click();
    await expect(page.getByPlaceholder("https://your-mcp-server.com")).toBeVisible({ timeout: 5000 });
    await page.getByPlaceholder("https://your-mcp-server.com").fill(MOCK_OAUTH_SERVER + "/mcp");
    await page.locator(".ant-select", { hasText: "Select auth type" }).click();
    await page.locator(".ant-select-dropdown:visible").getByText("OAuth").click();
    await expect(page.locator(".ant-select", { hasText: "Interactive (PKCE)" })).toBeVisible({ timeout: 5000 });
    await page.getByPlaceholder("https://example.com/oauth/authorize").fill(MOCK_OAUTH_SERVER + "/authorize");
    await page.getByPlaceholder("https://example.com/oauth/token").fill(MOCK_OAUTH_SERVER + "/token");

    // ---- click Authorize & Fetch Token -----------------------------------
    const authorizeBtn = page.getByRole("button", { name: "Authorize & Fetch Token" });
    await expect(authorizeBtn).toBeVisible({ timeout: 5000 });
    await authorizeBtn.click();

    // ---- wait for the full flow to complete ------------------------------
    // Chain: /authorize [A: intercepted] -> HTML writes sessionStorage + navigates
    //   -> back to mcp-servers -> resumeOAuthFlow fires -> POST /token [B: intercepted]
    //   -> "Token fetched."
    await expect(page.getByText(/Token fetched/)).toBeVisible({ timeout: 60000 });
  });
});
