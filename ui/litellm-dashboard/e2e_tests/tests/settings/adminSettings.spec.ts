import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";

test.describe("Add Model", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("admin settings test", async ({ page }) => {
    await page.goto("/ui");
    await page.getByRole("menuitem", { name: /Settings/ }).click();
    await page.getByRole("menuitem", { name: /Admin Settings/ }).click();
    await page.getByRole("tab", { name: "UI Settings" }).click();
    await expect(page.getByText("Configuration for UI-specific")).toBeVisible();
  });
});
