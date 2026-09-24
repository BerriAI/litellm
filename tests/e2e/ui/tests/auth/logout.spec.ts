import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

test.describe("Logout", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Clicking Logout clears the session and forces re-login on a protected page", async ({ page }) => {
    await page.goto("/ui");
    // Scope to the sidebar; the top-bar breadcrumb also shows "Virtual Keys".
    await expect(page.getByRole("complementary").getByText("Virtual Keys")).toBeVisible({ timeout: 10_000 });

    // Open the sidebar account menu. The trigger button exposes an aria-label
    // of "Account menu — <role> — signed in as <email>"; clicking it opens the
    // Base UI popover panel.
    await page.getByRole("button", { name: /Account menu/i }).click();

    const popup = page.getByTestId("sidebar-account-menu-panel");
    await expect(popup).toBeVisible({ timeout: 5_000 });

    // Click Logout — the handler clears the auth cookie and navigates via
    // window.location.href = PROXY_LOGOUT_URL (empty string in the e2e env).
    await popup.getByRole("button", { name: "Logout" }).click();

    // The cookie is now gone — visiting a protected page must redirect to /ui/login.
    await page.goto("/ui?page=llm-playground", { waitUntil: "domcontentloaded" });
    await expect(page).toHaveURL(/\/ui\/login/);
    await expect(page.getByRole("heading", { name: "Login" })).toBeVisible({ timeout: 10_000 });
  });
});
