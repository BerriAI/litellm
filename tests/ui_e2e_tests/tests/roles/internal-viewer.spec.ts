import { test, expect } from "@playwright/test";
import { Page, Role } from "../../constants";
import { loginAs } from "../../helpers/login";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Internal User Viewer Role", () => {
  test("Can only see allowed pages", async ({ page }) => {
    await loginAs(page, Role.InternalUserViewer);
    await expect(page.getByRole("menuitem", { name: "Virtual Keys" })).toBeVisible();
    await expect(page.getByRole("menuitem", { name: "Admin Settings" })).not.toBeVisible();
  });

  test("Cannot create keys", async ({ page }) => {
    await loginAs(page, Role.InternalUserViewer);
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByRole("button", { name: /Create New Key/i })).not.toBeVisible();
  });

  test("Cannot edit or delete keys", async ({ page }) => {
    await loginAs(page, Role.InternalUserViewer);
    await navigateToPage(page, Page.ApiKeys);
    // Ensure the keys table has loaded before asserting absence of actions
    await expect(page.getByRole("menuitem", { name: "Virtual Keys" })).toBeVisible();
    await expect(page.getByRole("button", { name: /Edit Key/i })).not.toBeVisible();
    await expect(page.getByRole("button", { name: /Delete Key/i })).not.toBeVisible();
    await expect(page.getByRole("button", { name: /Regenerate Key/i })).not.toBeVisible();
  });
});
