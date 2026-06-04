import { test, expect } from "@playwright/test";
import {
  E2E_INTERNAL_USER_EMAIL,
  E2E_INTERNAL_USER_ID,
  E2E_PROXY_ADMIN_EMAIL,
  E2E_PROXY_ADMIN_USER_ID,
  INTERNAL_USER_STORAGE_PATH,
} from "../../constants";

const escapeRegExp = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

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
      new RegExp(
        `signed in as (${escapeRegExp(E2E_INTERNAL_USER_EMAIL)}|${escapeRegExp(E2E_INTERNAL_USER_ID)})`,
      ),
      { timeout: 5_000 },
    );

    // Open the dropdown (UserDropdown configures trigger=["click"]).
    await accountButton.click();

    // Locate the panel by its test id (data-testid on the popupRender div in
    // UserDropdown.tsx) rather than Ant/Tailwind class names, so styling
    // refactors don't silently break the identity-scoping assertions below.
    const popup = page.getByTestId("user-dropdown-panel");
    await expect(popup).toBeVisible({ timeout: 5_000 });

    // The popup must show the internal user's identity — not the seeded
    // proxy admin's email/id, which would indicate a session/scope leak.
    await expect(popup.getByText(E2E_INTERNAL_USER_EMAIL)).toBeVisible({ timeout: 5_000 });
    await expect(popup.getByText(E2E_INTERNAL_USER_ID)).toBeVisible({ timeout: 5_000 });
    await expect(popup.getByText("Internal User", { exact: true })).toBeVisible({ timeout: 5_000 });
    await expect(popup.getByText(E2E_PROXY_ADMIN_EMAIL)).toHaveCount(0);
    await expect(popup.getByText(E2E_PROXY_ADMIN_USER_ID)).toHaveCount(0);
  });
});
