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
});
