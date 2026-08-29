import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

test.describe("Add Model", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("admin settings test", async ({ page }) => {
    await page.goto("/ui");
    // "Settings" is a collapsible group (button) in the sidebar; expand it, then
    // click the "Admin Settings" child link. Scope to the complementary landmark.
    const sidebar = page.getByRole("complementary");
    await sidebar.getByRole("button", { name: /Settings/ }).click();
    await sidebar.getByRole("link", { name: /Admin Settings/ }).click();
    await page.getByRole("tab", { name: "UI Settings" }).click();
    await expect(page.getByText("Configuration for UI-specific")).toBeVisible();
  });
});
