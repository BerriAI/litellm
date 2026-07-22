import { expect, test } from "@playwright/test";
import { users } from "../../fixtures/users";
import { Role } from "../../fixtures/roles";
import { PROXY_BASE_URL } from "../../constants";

test("user can log in", async ({ page }) => {
  await page.goto(`${PROXY_BASE_URL}/ui/login`);
  await page.getByPlaceholder("Enter your username").fill(users[Role.ProxyAdmin].email);
  await page.getByPlaceholder("Enter your password").fill(users[Role.ProxyAdmin].password);
  const loginButton = page.getByRole("button", { name: "Login", exact: true });
  await expect(loginButton).toBeEnabled();
  await loginButton.click();
  // Scope to the sidebar; the top-bar breadcrumb also shows "Virtual Keys".
  await expect(page.getByRole("complementary").getByText("Virtual Keys")).toBeVisible();

  // Match the sidebar account button by its stable aria-label
  // (SidebarAccountMenu emits "Account menu — <role> — signed in as <email|id>";
  // displayName is "Account" for the master-key admin, so match on the label).
  const userTrigger = page.locator('button[aria-label^="Account menu"]').first();
  await userTrigger.click();

  // The account menu is a Base UI popover; locate its panel by test id.
  const popup = page.getByTestId("sidebar-account-menu-panel");
  await expect(popup).toBeVisible({ timeout: 5_000 });
  await expect(popup.getByText("Admin", { exact: true })).toBeVisible({ timeout: 5_000 });
  await expect(popup.getByText("default_user_id", { exact: true })).toBeVisible({ timeout: 5_000 });
});
