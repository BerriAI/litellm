import { test, expect } from "@playwright/test";
import { INTERNAL_USER_STORAGE_PATH } from "../../constants";

test.describe("Navbar identity scoping", () => {
  test.use({ storageState: INTERNAL_USER_STORAGE_PATH });

  test("Internal user navbar dropdown shows their own role and user id, not the admin's", async ({ page }) => {
    await page.goto("/ui");
    await expect(page.getByText("Virtual Keys")).toBeVisible({ timeout: 10_000 });

    // The account menu button carries the user's role and email/id in its
    // aria-label (see UserDropdown.tsx). Match by partial role.
    const accountButton = page.locator('button[aria-label^="Account menu"]').first();
    await expect(accountButton).toHaveAttribute("aria-label", /Internal User/, { timeout: 5_000 });
    await expect(accountButton).toHaveAttribute(
      "aria-label",
      /signed in as internal@test\.local|signed in as e2e-internal-user/,
      { timeout: 5_000 },
    );

    // Open the dropdown (UserDropdown configures trigger=["click"]).
    await accountButton.click();

    const popup = page.locator(".ant-dropdown:visible").filter({
      has: page.locator(".bg-white.rounded-lg.shadow-lg"),
    }).first();
    await expect(popup).toBeVisible({ timeout: 5_000 });

    // The popup must show the internal user's identity — not the seeded
    // proxy admin's email/id, which would indicate a session/scope leak.
    await expect(popup.getByText("internal@test.local")).toBeVisible({ timeout: 5_000 });
    await expect(popup.getByText("e2e-internal-user")).toBeVisible({ timeout: 5_000 });
    await expect(popup.getByText("Internal User", { exact: true })).toBeVisible({ timeout: 5_000 });
    await expect(popup.getByText("admin@test.local")).toHaveCount(0);
    await expect(popup.getByText("e2e-proxy-admin")).toHaveCount(0);
  });
});
