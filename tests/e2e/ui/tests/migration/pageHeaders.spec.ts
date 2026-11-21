import { test, expect } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { MIGRATED_E2E_PAGES } from "../../fixtures/migratedPages";
import { dismissFeedbackPopup, expectUiRoute } from "../../helpers/navigation";

const ROOT = (process.env.SERVER_ROOT_PATH ?? "").replace(/\/+$/, "");
const API_PLAYGROUND_ROUTE = "transform-request";

const routes = [
  ...Object.values(MIGRATED_E2E_PAGES)
    .map(({ segment }) => segment)
    .filter((segment) => segment !== API_PLAYGROUND_ROUTE),
  "home",
  "alerts",
  "customers",
  "apps",
  "customers/access",
  "customers/settings",
];

test.use({ storageState: ADMIN_STORAGE_PATH });

test.describe("Dashboard page headers", () => {
  for (const route of routes) {
    test(`${route} renders its local heading in PageHeader`, async ({
      page,
    }) => {
      await page.goto(`${ROOT}/ui/${route}`);
      await expectUiRoute(page, route);
      await dismissFeedbackPopup(page);
      const header = page
        .getByRole("main")
        .locator('[data-slot="page-header"]');
      await expect(header).toHaveCount(1);
      await expect(header.getByRole("heading", { level: 1 })).toBeVisible();
    });
  }
});
