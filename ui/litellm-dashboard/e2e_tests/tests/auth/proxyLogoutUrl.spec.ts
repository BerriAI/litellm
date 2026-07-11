import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

/**
 * Runs as part of the standard e2e suite: both `run_e2e.sh` and the CircleCI
 * `e2e_ui_testing` job boot the proxy with PROXY_LOGOUT_URL=https://www.example.com
 * and export the same value to this Playwright process. The spec reads it to
 * know where the browser is expected to land.
 *
 * The skip guard below is a safety net for environments that launch the proxy
 * without the env var (e.g. an ad-hoc `npx playwright test` against a default
 * proxy) — there the logout target is empty and this contract can't be checked.
 */
const LOGOUT_URL = process.env.PROXY_LOGOUT_URL ?? "";

test.skip(!LOGOUT_URL, "Requires PROXY_LOGOUT_URL env var");

test.describe("PROXY_LOGOUT_URL redirect", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Logout clears the session and redirects to PROXY_LOGOUT_URL", async ({ page }) => {
    const target = new URL(LOGOUT_URL);

    // Stub the external logout destination so the assertion doesn't depend on
    // that host being reachable from CI — we only care that the browser is sent
    // there, not what it serves back.
    await page.route(
      (url) => url.origin === target.origin,
      (route) =>
        route.fulfill({
          status: 200,
          contentType: "text/html",
          body: "<html><body>logged out</body></html>",
        }),
    );

    // navbar.tsx populates the logout target only after the proxy UI settings
    // fetch (/sso/get/ui_settings) resolves. Clicking Logout before that lands
    // runs `window.location.href = ""` — a same-origin reload, not a redirect —
    // so gate the click on the settings response, not just on first paint.
    const settingsLoaded = page.waitForResponse((r) => r.url().includes("/sso/get/ui_settings") && r.ok(), {
      timeout: 30_000,
    });
    await page.goto("/ui");
    // Scope to the sidebar; the top-bar breadcrumb also shows "Virtual Keys".
    await expect(page.getByRole("complementary").getByText("Virtual Keys")).toBeVisible({ timeout: 15_000 });
    await settingsLoaded;

    // Pre-condition: we start authenticated. The admin storage state carries a
    // `token` cookie, so a real logout has something to tear down.
    const tokensBefore = (await page.context().cookies()).filter((c) => c.name === "token");
    expect(tokensBefore.length, "should start logged in with a token cookie").toBeGreaterThan(0);

    // Open the navbar account dropdown (trigger=click) and click Logout by role
    // rather than internal Ant Design CSS classes, which are not a stable API.
    await page.getByRole("button", { name: /^Account menu/ }).click();
    const logout = page.getByRole("menuitem", { name: "Logout" });
    await expect(logout).toBeVisible({ timeout: 5_000 });

    // handleLogout clears cookies/local storage, then assigns window.location.href.
    // Arm the navigation wait before the click so we never miss the redirect.
    await Promise.all([page.waitForURL((url) => url.origin === target.origin, { timeout: 15_000 }), logout.click()]);

    // The browser landed on exactly the configured logout URL. Compare normalized
    // hrefs (both sides through URL()) so trailing-slash / default-port rewrites the
    // browser applies are matched on the expected side too — this pins scheme, host,
    // port, path, query and hash, not just the origin.
    const landed = new URL(page.url());
    expect(landed.href).toBe(target.href);

    // ...and the client-side session cookie is gone (clearTokenCookies ran before
    // the redirect). HttpOnly cookies set server-side can't be cleared from JS,
    // so scope the check to the JS-managed token the UI is responsible for.
    const clientTokensAfter = (await page.context().cookies()).filter((c) => c.name === "token" && !c.httpOnly);
    expect(clientTokensAfter, "client token cookie should be cleared on logout").toHaveLength(0);
  });
});
