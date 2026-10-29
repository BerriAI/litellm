import { test, expect, type Page } from "@playwright/test";
import { MIGRATED_E2E_PAGES, type MigratedPage } from "../../fixtures/migratedPages";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { clickSidebarLink, dismissFeedbackPopup, expectUiRoute, sidebarLink } from "../../helpers/navigation";
import { proxyIsPremium } from "../../helpers/premium";
import { masterKey } from "../../helpers/traffic";

const ROOT = (process.env.SERVER_ROOT_PATH ?? "").replace(/\/+$/, "");
const apiKeys = MIGRATED_E2E_PAGES["api-keys"];
const home: MigratedPage = { segment: "", linkName: "Home", content: { role: "heading", name: "Home" } };

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
  test("keeps the configured favicon after navigating from UI Theme to Home", async ({ page, request }) => {
    const themeResponse = await request.get(`${ROOT}/get/ui_theme_settings`);
    expect(themeResponse.ok()).toBe(true);
    const originalTheme: { values: Record<string, string | null> } = await themeResponse.json();

    await page.goto(`${ROOT}/ui/ui-theme/`);
    const faviconInput = page.getByLabel("Custom Favicon URL", { exact: true });
    await expect(faviconInput).toHaveValue(originalTheme.values.favicon_url ?? "");

    const customIconUrl = new URL(`${ROOT}/ui/next.svg`, page.url()).href;
    const customIcon = await request.get(customIconUrl);
    expect(customIcon.ok()).toBe(true);
    const customIconBody = await customIcon.body();

    try {
      await faviconInput.fill(customIconUrl);
      const saved = page.waitForResponse(
        (response) =>
          new URL(response.url()).pathname === `${ROOT}/update/ui_theme_settings` &&
          response.request().method() === "PATCH",
      );
      await page.getByRole("button", { name: "Save Changes", exact: true }).click();
      expect((await saved).ok()).toBe(true);

      await page.reload();
      await expect(faviconInput).toHaveValue(customIconUrl);
      await clickSidebarLink(page, "Home");
      await expectUiRoute(page, "home");
      await expect(page.getByRole("heading", { name: "Home", exact: true })).toBeVisible();

      await expect
        .poll(() =>
          page
            .locator('link[rel="icon"]')
            .evaluateAll(
              (icons, url) => icons.length > 0 && icons.every((icon) => (icon as HTMLLinkElement).href === url),
              customIconUrl,
            ),
        )
        .toBe(true);
      const iconUrls = await page
        .locator('link[rel="icon"]')
        .evaluateAll((icons) => icons.map((icon) => (icon as HTMLLinkElement).href));
      expect(iconUrls.length).toBeGreaterThan(0);
      for (const iconUrl of iconUrls) {
        const icon = await request.get(iconUrl);
        expect(icon.ok(), `Favicon ${iconUrl} loads after navigation`).toBe(true);
        expect(await icon.body(), `Favicon ${iconUrl} uses the saved image`).toEqual(customIconBody);
      }
    } finally {
      const restored = await request.patch(`${ROOT}/update/ui_theme_settings`, {
        headers: { Authorization: `Bearer ${masterKey()}` },
        data: originalTheme.values,
      });
      expect(restored.ok(), "Restore the original theme settings").toBe(true);
    }
  });

  for (const destination of Object.values(MIGRATED_E2E_PAGES)) {
    test(`${destination.segment}: sidebar nav, reload, and round-trip via the api-keys landing`, async ({ page }) => {
      const pageErrors: string[] = [];
      page.on("pageerror", (error) => pageErrors.push(String(error)));

      const landing = await page.goto(`${ROOT}/ui/`);
      expect(landing?.ok(), "dashboard document loads successfully").toBe(true);
      await dismissFeedbackPopup(page);
      await expectContent(page, home);

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
    await expectContent(page, home);

    for (const destination of [apiKeys, MIGRATED_E2E_PAGES.models, apiKeys]) {
      await navigateToDestination(page, destination);
    }
    expect(pageErrors, "page errors during migrated page navigation").toEqual([]);
  });
});
