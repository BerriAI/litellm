import { test, expect } from "@playwright/test";
import { Page, Role } from "../../constants";
import { loginAs } from "../../helpers/login";
import { navigateToPage } from "../../helpers/navigation";

test.describe("Team Admin Role", () => {
  test("Can view team keys but not admin settings", async ({ page }) => {
    await loginAs(page, Role.TeamAdmin);
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByRole("menuitem", { name: "Virtual Keys" })).toBeVisible();
    await expect(page.getByRole("menuitem", { name: "Admin Settings" })).not.toBeVisible();
  });
});
