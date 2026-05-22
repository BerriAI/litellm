import { expect, test } from "@playwright/test";
import { users } from "../../fixtures/users";
import { Role } from "../../fixtures/roles";

test("user can log in", async ({ page }) => {
  await page.goto("http://localhost:4000/ui/login");
  await page.getByPlaceholder("Enter your username").fill(users[Role.ProxyAdmin].email);
  await page.getByPlaceholder("Enter your password").fill(users[Role.ProxyAdmin].password);
  const loginButton = page.getByRole("button", { name: "Login", exact: true });
  await expect(loginButton).toBeEnabled();
  await loginButton.click();
  await expect(page.getByText("Virtual Keys")).toBeVisible();

  // Dispatch hover events directly — antd Dropdown's default hover trigger
  // closes the popup as soon as the cursor moves off the button, which
  // happens between Playwright assertions.
  const userTrigger = page.locator("nav").getByRole("button").filter({ hasText: /^User$/ });
  await userTrigger.evaluate((el: HTMLElement) => {
    el.dispatchEvent(new MouseEvent("mouseover", { bubbles: true }));
    el.dispatchEvent(new MouseEvent("mouseenter", { bubbles: true }));
  });

  // Filter by the popupRender wrapper class to disambiguate from other
  // ant-dropdown popups.
  const popup = page.locator(".ant-dropdown:visible").filter({
    has: page.locator(".bg-white.rounded-lg.shadow-lg"),
  }).first();
  await expect(popup).toBeVisible({ timeout: 5_000 });
  await expect(popup.getByText("Admin", { exact: true })).toBeVisible({ timeout: 5_000 });
  await expect(popup.getByText("default_user_id", { exact: true })).toBeVisible({ timeout: 5_000 });
});
