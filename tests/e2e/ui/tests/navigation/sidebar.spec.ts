import test, { expect } from "@playwright/test";
import { Role } from "../../fixtures/roles";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { menuLabelToPage } from "../../fixtures/menuMappings";
import { navigateToPage } from "../../helpers/navigation";
import { MIGRATED_E2E_PAGES } from "../../fixtures/migratedPages";
import type { Page as PlaywrightPage } from "@playwright/test";

const sidebarButtons = {
  [Role.ProxyAdmin]: [
    "Virtual Keys",
    "Playground",
    "Models",
    "Usage",
    "Teams",
    "Internal Users",
    "AI Hub",
    "Response Cache",
  ],
};

/** Migrated pages live at a path route; legacy pages keep the ?page= query param. */
async function expectPageUrl(page: PlaywrightPage, pageKey: string): Promise<void> {
  const migratedSegment = MIGRATED_E2E_PAGES[pageKey];
  if (migratedSegment) {
    await expect(page).toHaveURL(new RegExp(`/ui/${migratedSegment}/?($|\\?)`));
  } else {
    await expect(page).toHaveURL(new RegExp(`[?&]page=${pageKey}(&|$)`));
  }
}

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

        // Sidebar items are links inside the `complementary` landmark; scoping
        // there avoids the top-bar breadcrumb, which also links the page name.
        const tab = page.getByRole("complementary").getByRole("link", { name: buttonLabel });
        await expect(tab).toBeVisible();

        await tab.click();

        await expectPageUrl(page, expectedPage);
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
      await expectPageUrl(page, Page.ApiKeys);

      await navigateToPage(page, Page.Models);
      await expectPageUrl(page, Page.Models);

      // Migrated page: /ui?page=llm-playground redirects to the path route
      await navigateToPage(page, Page.LlmPlayground);
      await expectPageUrl(page, Page.LlmPlayground);
    });
  });
}
