import { test, expect } from "@playwright/test";

test("login and save auth state", async ({ page }) => {
  await page.goto("http://localhost:4000/ui");

  await page.getByPlaceholder("Enter your username").fill("admin");
  await page.getByPlaceholder("Enter your password").fill("gm");

  const loginButton = page.getByRole("button", { name: "Login" });
  await expect(loginButton).toBeEnabled();
  await loginButton.click();

  // Assert successful login (important)
  await expect(page.getByText("AI Gateway")).toBeVisible();

  // 🔐 Save auth state
  await page.context().storageState({
    path: "storageState.json",
  });
});
