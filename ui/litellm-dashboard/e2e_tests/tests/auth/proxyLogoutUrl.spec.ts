import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

/**
 * Env-gated: only runs when the proxy is launched with PROXY_LOGOUT_URL set.
 *
 * `run_e2e.sh` deliberately exports PROXY_LOGOUT_URL="" so the rest of the suite
 * gets the default behaviour. To exercise this contract, re-launch the proxy
 * with the env var and re-run this spec, e.g.:
 *
 *   PROXY_LOGOUT_URL=https://www.example.com ./run_e2e.sh -g "PROXY_LOGOUT_URL"
 *
 * Pattern matches the existing env-gated `serverRootPathRedirect.spec.ts`.
 */
const LOGOUT_URL = process.env.PROXY_LOGOUT_URL ?? "";

test.skip(!LOGOUT_URL, "Requires PROXY_LOGOUT_URL env var");

test.describe("PROXY_LOGOUT_URL redirect", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Logout sends the user to PROXY_LOGOUT_URL", async ({ page }) => {
    await page.goto("/ui");
    await expect(page.getByText("Virtual Keys")).toBeVisible({ timeout: 10_000 });

    // Open the navbar account dropdown (UserDropdown uses trigger=click)
    const accountButton = page.locator('button[aria-label^="Account menu"]').first();
    await accountButton.click();

    const popup = page.locator(".ant-dropdown:visible").filter({
      has: page.locator(".bg-white.rounded-lg.shadow-lg"),
    }).first();
    await expect(popup).toBeVisible({ timeout: 5_000 });

    // After clicking Logout, navbar.tsx assigns window.location.href = LOGOUT_URL.
    // Wait for navigation to the external host instead of a same-origin reload.
    await Promise.all([
      page.waitForURL(new RegExp(LOGOUT_URL.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")), {
        timeout: 15_000,
      }),
      popup.getByText("Logout", { exact: true }).click(),
    ]);

    expect(page.url()).toContain(LOGOUT_URL);
  });
});
