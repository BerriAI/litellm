import test, { expect } from "@playwright/test";
import { Role } from "../../fixtures/roles";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { menuLabelToPage } from "../../fixtures/menuMappings";
import {
  clickSidebarLink,
  dismissFeedbackPopup,
  expectUiRoute,
  navigateToPage,
  sidebarLink,
} from "../../helpers/navigation";
import { MIGRATED_E2E_PAGES } from "../../fixtures/migratedPages";
import type { Page as PlaywrightPage } from "@playwright/test";

const sidebarButtons = {
  [Role.ProxyAdmin]: [
    "Virtual Keys",
    "Playground",
    "Models + Endpoints",
    "Usage",
    "Teams",
    "Internal Users",
    "AI Hub",
    "Response Cache",
  ],
};

/** Migrated pages live at a path route; legacy pages keep the ?page= query param. */
async function expectPageUrl(page: PlaywrightPage, pageKey: string): Promise<void> {
  const migratedPage = MIGRATED_E2E_PAGES[pageKey];
  if (migratedPage) {
    await expectUiRoute(page, migratedPage.segment);
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

        await clickSidebarLink(page, buttonLabel);

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

    for (const format of ["without trailing slash", "absolute with query and fragment", "relative"] as const) {
      test(`sidebar locator tolerates hrefs ${format}`, async ({ page }) => {
        await page.goto("/ui/");
        await dismissFeedbackPopup(page);
        const link = sidebarLink(page, "Models + Endpoints");
        await expect(link).toBeVisible();
        const destination = new URL("/ui/models-and-endpoints/", page.url());
        const href =
          format === "without trailing slash"
            ? destination.pathname.replace(/\/$/, "")
            : format === "relative"
              ? "./models-and-endpoints/"
              : `${destination.href}?source=navigation-smoke#overview`;

        await link.evaluate((element, value) => element.setAttribute("href", value), href);
        await expect(link).toHaveAttribute("href", href);
        await clickSidebarLink(page, "Models + Endpoints");

        await expectUiRoute(page, "models-and-endpoints");
        await expect(
          page.getByRole("main").getByRole("heading", { name: "Model Management", exact: true }),
        ).toBeVisible();
      });
    }

    test("route assertion tolerates a query string and fragment on a deep link", async ({ page }) => {
      const response = await page.goto("/ui/models-and-endpoints/?source=navigation-smoke#overview");
      expect(response?.ok()).toBe(true);
      await expectUiRoute(page, "models-and-endpoints");
      await expect(
        page.getByRole("main").getByRole("heading", { name: "Model Management", exact: true }),
      ).toBeVisible();
      expect(new URL(page.url()).search).toBe("?source=navigation-smoke");
      expect(new URL(page.url()).hash).toBe("#overview");
    });
  });
}
