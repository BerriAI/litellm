import { Page as PlaywrightPage, expect } from "@playwright/test";
import { users, Role } from "../constants";

export async function loginAs(page: PlaywrightPage, role: Role) {
  const user = users[role];
  await page.goto("/ui/login");
  await page.getByPlaceholder("Enter your username").fill(user.email);
  await page.getByPlaceholder("Enter your password").fill(user.password);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  // Wait for navigation away from login page into the dashboard
  await page.waitForURL((url) => url.pathname.startsWith("/ui") && !url.pathname.includes("/login"), {
    timeout: 30_000,
  });
  // Wait for sidebar to render as a signal that the dashboard is ready
  await expect(page.getByRole("menuitem", { name: "Virtual Keys" })).toBeVisible({ timeout: 30_000 });
}
