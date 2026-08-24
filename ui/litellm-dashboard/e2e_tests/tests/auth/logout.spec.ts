import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

test.describe("Logout", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Clicking Logout clears the session and forces re-login on a protected page", async ({ page }) => {
    await page.goto("/ui");
    await expect(page.getByText("Virtual Keys")).toBeVisible({ timeout: 10_000 });

    // Open the navbar User dropdown. The trigger button exposes an aria-label
    // of "Account menu — <role> — signed in as <email>", and the antd Dropdown
    // is declared with trigger={["click"]}, so a plain click opens the popup.
    await page.getByRole("button", { name: /Account menu/i }).click();

    const popup = page
      .locator(".ant-dropdown:visible")
      .filter({
        has: page.locator(".bg-white.rounded-lg.shadow-lg"),
      })
      .first();
    await expect(popup).toBeVisible({ timeout: 5_000 });

    // Click Logout — the handler clears the auth cookie and navigates via
    // window.location.href = PROXY_LOGOUT_URL (empty string in the e2e env).
    await popup.getByText("Logout", { exact: true }).click();

    // The cookie is now gone — visiting a protected page must redirect to /ui/login.
    await page.goto("/ui?page=llm-playground", { waitUntil: "domcontentloaded" });
    await expect(page).toHaveURL(/\/ui\/login/);
    await expect(page.getByRole("heading", { name: "Login" })).toBeVisible({ timeout: 10_000 });
  });
});
