import test, { expect } from "@playwright/test";
import { Role } from "../../fixtures/roles";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { menuLabelToPage } from "../../fixtures/menuMappings";
import { navigateToPage } from "../../helpers/navigation";

const sidebarButtons = {
  [Role.ProxyAdmin]: [
    "Virtual Keys",
    "Playground",
    "Models",
    "Usage",
    "Teams",
    "Internal Users",
    "API Reference",
    "AI Hub",
  ],
};

const roles = [{ role: Role.ProxyAdmin, storage: ADMIN_STORAGE_PATH }];

for (const { role, storage } of roles) {
  test.describe(`${role} sidebar`, () => {
    test.use({ storageState: storage });

    test("should navigate to correct URL when clicking sidebar menu items from homepage", async ({ page }) => {
      await page.goto("/ui");
      await page.evaluate(() => {
        window.localStorage.setItem("disableUsageIndicator", "true");
        window.localStorage.setItem("disableShowPrompts", "true");
        window.localStorage.setItem("disableShowNewBadge", "true");
      });

      for (const buttonLabel of sidebarButtons[role as keyof typeof sidebarButtons]) {
        const expectedPage = menuLabelToPage[buttonLabel];

        if (!expectedPage) {
          throw new Error(`No page mapping found for menu label: ${buttonLabel}`);
        }

        const tab = page.getByRole("menuitem", { name: buttonLabel });
        await expect(tab).toBeVisible();

        await tab.click();

        // Verify URL contains the correct page query parameter
        await expect(page).toHaveURL(new RegExp(`[?&]page=${expectedPage}(&|$)`));
      }
    });

    test("should navigate directly to page using navigation helper", async ({ page }) => {
      await page.goto("/ui");
      await page.evaluate(() => {
        window.localStorage.setItem("disableUsageIndicator", "true");
        window.localStorage.setItem("disableShowPrompts", "true");
        window.localStorage.setItem("disableShowNewBadge", "true");
      });

      // Test direct navigation to verify the helper function works
      await navigateToPage(page, Page.ApiKeys);
      await expect(page).toHaveURL(new RegExp(`[?&]page=${Page.ApiKeys}(&|$)`));

      await navigateToPage(page, Page.Models);
      await expect(page).toHaveURL(new RegExp(`[?&]page=${Page.Models}(&|$)`));

      await navigateToPage(page, Page.LlmPlayground);
      await expect(page).toHaveURL(new RegExp(`[?&]page=${Page.LlmPlayground}(&|$)`));
    });
  });
}
