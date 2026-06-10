import test, { expect } from "@playwright/test";
import { Role } from "../../fixtures/roles";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { menuLabelToPage } from "../../fixtures/menuMappings";
import { navigateToPage } from "../../helpers/navigation";

const sidebarButtons = {
  [Role.ProxyAdmin]: ["Virtual Keys", "Playground", "Models", "Usage", "Teams", "Internal Users", "AI Hub"],
};

// Route segment for pages migrated to path routes; mirror of MIGRATED_PAGES in src/utils/migratedPages.ts.
const migratedPageSegments: Partial<Record<Page, string>> = {
  [Page.ApiRef]: "api-reference",
  [Page.LlmPlayground]: "playground",
};

function expectedUrlPattern(pageKey: Page): RegExp {
  const segment = migratedPageSegments[pageKey];
  return segment ? new RegExp(`/ui/${segment}/?($|\\?)`) : new RegExp(`[?&]page=${pageKey}(&|$)`);
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

        const tab = page.getByRole("menuitem", { name: buttonLabel });
        await expect(tab).toBeVisible();

        await tab.click();

        await expect(page).toHaveURL(expectedUrlPattern(expectedPage));
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
      await expect(page).toHaveURL(expectedUrlPattern(Page.ApiKeys));

      await navigateToPage(page, Page.Models);
      await expect(page).toHaveURL(expectedUrlPattern(Page.Models));

      await navigateToPage(page, Page.LlmPlayground);
      await expect(page).toHaveURL(expectedUrlPattern(Page.LlmPlayground));
    });
  });
}
