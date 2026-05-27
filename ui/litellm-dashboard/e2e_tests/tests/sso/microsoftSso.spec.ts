import { test, expect } from "@playwright/test";

/**
 * Env-gated: only runs when the proxy is launched with Microsoft SSO env vars
 * set (MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, MICROSOFT_TENANT). The
 * standard run_e2e.sh leaves these unset; this spec self-skips otherwise.
 *
 * The contract checked here is the UI-side gate: when SSO is configured on the
 * proxy, the login page exposes an enabled "Login with SSO" button that
 * navigates to `/sso/key/generate`, which is where the proxy starts the
 * Microsoft OAuth flow. The full Microsoft login flow (entering credentials,
 * MFA, etc.) is intentionally NOT automated — that requires a real test tenant
 * and brittle screen scraping of login.microsoftonline.com. To run the full
 * flow manually, set MICROSOFT_TEST_EMAIL + MICROSOFT_TEST_PASSWORD and exercise
 * it interactively against the live proxy.
 */
const MS_CLIENT_ID = process.env.MICROSOFT_CLIENT_ID ?? "";

test.skip(!MS_CLIENT_ID, "Requires MICROSOFT_CLIENT_ID env var (Microsoft SSO not configured)");

test.describe("Microsoft SSO", () => {
  test("Login page exposes an enabled SSO button that routes to /sso/key/generate", async ({ page }) => {
    await page.context().clearCookies();
    await page.goto("/ui/login");

    // The Login with SSO button is rendered enabled only when the proxy reports
    // `sso_configured: true` in its UI config (which it does when Microsoft env
    // is set). When SSO is not configured, the button is replaced by a tooltip-
    // wrapped disabled version.
    const ssoButton = page.getByRole("button", { name: /Login with SSO/i });
    await expect(ssoButton).toBeVisible({ timeout: 10_000 });
    await expect(ssoButton).toBeEnabled();

    // Clicking it should navigate to /sso/key/generate (the proxy then
    // bounces to login.microsoftonline.com). We assert on the proxy hop;
    // chasing the redirect to Microsoft's host would couple the test to
    // an external login UI that changes without warning.
    await Promise.all([
      page.waitForURL(/\/sso\/key\/generate/, { timeout: 15_000 }),
      ssoButton.click(),
    ]);

    expect(page.url()).toContain("/sso/key/generate");
  });
});
