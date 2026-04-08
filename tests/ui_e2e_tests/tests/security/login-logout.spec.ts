import { test, expect } from "@playwright/test";
import { users, Role } from "../../constants";

test.describe("Authentication", () => {
  test("Login with valid admin credentials", async ({ page }) => {
    await page.goto("/ui/login");
    await page.getByPlaceholder("Enter your username").fill(users[Role.ProxyAdmin].email);
    await page.getByPlaceholder("Enter your password").fill(users[Role.ProxyAdmin].password);
    await page.getByRole("button", { name: "Login", exact: true }).click();
    await expect(page.getByRole("menuitem", { name: "Virtual Keys" })).toBeVisible();
  });

  test("Unauthenticated user is redirected to login", async ({ page }) => {
    await page.goto("/ui");
    await page.waitForURL(/\/ui\/login/);
    await expect(page.getByRole("heading", { name: /Login/i })).toBeVisible();
  });
});
