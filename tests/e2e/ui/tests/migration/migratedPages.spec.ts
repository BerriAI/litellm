import { test, expect, type Page } from "@playwright/test";
import { MIGRATED_E2E_PAGES, type MigratedPage } from "../../fixtures/migratedPages";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { clickSidebarLink, dismissFeedbackPopup, expectUiRoute, sidebarLink } from "../../helpers/navigation";
import { proxyIsPremium } from "../../helpers/premium";

const ROOT = (process.env.SERVER_ROOT_PATH ?? "").replace(/\/+$/, "");
const apiKeys = MIGRATED_E2E_PAGES["api-keys"];

async function expectContent(page: Page, destination: MigratedPage): Promise<void> {
  await expect(sidebarLink(page, apiKeys.linkName)).toBeVisible({ timeout: 20_000 });
  const main = page.getByRole("main");
  if (destination.unlicensedText && !proxyIsPremium()) {
    await expect(main.getByText(destination.unlicensedText, { exact: true })).toBeVisible();
    return;
  }
  const content = destination.content;
  const landmark =
    "role" in content
      ? main.getByRole(content.role, { name: content.name, exact: true })
      : main.getByText(content.text, { exact: true });
  await expect(landmark).toBeVisible();
}

async function navigateToDestination(page: Page, destination: MigratedPage): Promise<void> {
  await clickSidebarLink(page, destination.linkName, destination.group);
  await expectUiRoute(page, destination.segment);
  await dismissFeedbackPopup(page);
  await expectContent(page, destination);
}

test.use({ storageState: ADMIN_STORAGE_PATH });

test.describe("App Router migrated pages", () => {
  for (const destination of Object.values(MIGRATED_E2E_PAGES)) {
    test(`${destination.segment}: sidebar nav, reload, and round-trip via the api-keys landing`, async ({ page }) => {
      const pageErrors: string[] = [];
      page.on("pageerror", (error) => pageErrors.push(String(error)));

      const landing = await page.goto(`${ROOT}/ui/`);
      expect(landing?.ok(), "dashboard document loads successfully").toBe(true);
      await dismissFeedbackPopup(page);
      await expectContent(page, apiKeys);

      await navigateToDestination(page, destination);

      const reloaded = await page.reload();
      expect(reloaded?.ok(), `${destination.segment} document loads on reload`).toBe(true);
      await expectUiRoute(page, destination.segment);
      await dismissFeedbackPopup(page);
      await expectContent(page, destination);

      await navigateToDestination(page, apiKeys);
      await navigateToDestination(page, destination);
      expect(pageErrors, `page errors during ${destination.segment} journey`).toEqual([]);
    });
  }

  test("navigates directly between two migrated pages", async ({ page }) => {
    const pageErrors: string[] = [];
    page.on("pageerror", (error) => pageErrors.push(String(error)));

    const landing = await page.goto(`${ROOT}/ui/`);
    expect(landing?.ok(), "dashboard document loads successfully").toBe(true);
    await dismissFeedbackPopup(page);
    await expectContent(page, apiKeys);

    for (const destination of [apiKeys, MIGRATED_E2E_PAGES.models, apiKeys]) {
      await navigateToDestination(page, destination);
    }
    expect(pageErrors, "page errors during migrated page navigation").toEqual([]);
  });
});
