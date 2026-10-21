import { test, expect } from "@playwright/test";
import { Role, users } from "../../fixtures/users";
import { logInThroughLoginPage } from "../../helpers/userOnboarding";

const LOGOUT_URL = process.env.PROXY_LOGOUT_URL ?? "";

test.describe("Logout", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("Clicking Logout clears the session and forces re-login on a protected page", async ({ page }) => {
    const admin = users[Role.ProxyAdmin];
    await logInThroughLoginPage(page, admin.email, admin.password);

    if (LOGOUT_URL) {
      const target = new URL(LOGOUT_URL);
      await page.route(
        (url) => url.origin === target.origin,
        (route) =>
          route.fulfill({ status: 200, contentType: "text/html", body: "<html><body>logged out</body></html>" }),
      );
    }
    const settingsLoaded = page.waitForResponse((r) => r.url().includes("/sso/get/ui_settings") && r.ok(), {
      timeout: 30_000,
    });
    await page.goto("/ui");
    // Scope to the sidebar; the top-bar breadcrumb also shows "Virtual Keys".
    await expect(page.getByRole("complementary").getByText("Virtual Keys")).toBeVisible({ timeout: 10_000 });
    await settingsLoaded;

    // Open the sidebar account menu. The trigger button exposes an aria-label
    // of "Account menu — <role> — signed in as <email>"; clicking it opens the
    // Base UI popover panel.
    await page.getByRole("button", { name: /Account menu/i }).click();

    const popup = page.getByTestId("sidebar-account-menu-panel");
    await expect(popup).toBeVisible({ timeout: 5_000 });

    // Click Logout — the handler clears the auth cookie and navigates via
    // window.location.href = PROXY_LOGOUT_URL.
    const loggedOut = LOGOUT_URL
      ? page.waitForURL((url) => url.origin === new URL(LOGOUT_URL).origin, { timeout: 15_000 })
      : page.waitForEvent("load", { timeout: 15_000 });
    await popup.getByRole("button", { name: "Logout" }).click();
    await loggedOut;
    await expect
      .poll(async () => (await page.context().cookies()).filter((c) => c.name === "token").length, {
        timeout: 15_000,
      })
      .toBe(0);

    // The cookie is now gone — visiting a protected page must redirect to /ui/login.
    await page.goto("/ui?page=llm-playground", { waitUntil: "domcontentloaded" });
    await expect(page).toHaveURL(/\/ui\/login/);
    await expect(page.getByRole("heading", { name: "Login" })).toBeVisible({ timeout: 10_000 });
  });
});
