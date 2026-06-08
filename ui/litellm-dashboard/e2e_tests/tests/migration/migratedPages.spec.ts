import { test, expect } from "@playwright/test";
import { MIGRATED_E2E_SEGMENTS } from "../../fixtures/migratedPages";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { dismissFeedbackPopup } from "../../helpers/navigation";

/**
 * App Router migration smoke. For each migrated page we deep-link to its path
 * route and then navigate away, asserting no failure in either situation. Driven
 * by MIGRATED_E2E_SEGMENTS, so it grows as pages are migrated.
 *
 * SERVER_ROOT_PATH (e.g. "/litellm") exercises the non-root mount; leave it unset
 * for the default mount. Boot the proxy with the matching value before running.
 */
const ROOT = process.env.SERVER_ROOT_PATH ?? "";
// Optional: linger on each state so a human can watch a headed run. No-op (0) by default.
const WATCH_MS = Number(process.env.E2E_WATCH_MS ?? 0);
const watch = (page: import("@playwright/test").Page) => (WATCH_MS ? page.waitForTimeout(WATCH_MS) : Promise.resolve());

test.use({ storageState: ADMIN_STORAGE_PATH });

test.describe("App Router migrated pages", () => {
  for (const segment of MIGRATED_E2E_SEGMENTS) {
    test(`${segment}: deep-links to its path route and navigates away cleanly`, async ({ page }) => {
      const pageErrors: string[] = [];
      page.on("pageerror", (e) => pageErrors.push(String(e)));

      // 1. The migrated page is served at its own path (this is what a wrong
      //    server_root_path breaks: the static export must serve <root>/ui/<segment>).
      await page.goto(`${ROOT}/ui/${segment}`);
      await dismissFeedbackPopup(page);

      await expect(page).toHaveURL(new RegExp(`${ROOT}/ui/${segment}/?($|\\?)`));
      // The dashboard shell rendered and the route did not 404 / crash.
      await expect(page.locator("a", { hasText: "Virtual Keys" })).toBeVisible({ timeout: 20_000 });
      expect(pageErrors, `page errors on ${ROOT}/ui/${segment}`).toEqual([]);
      await watch(page);

      // 2. Clicking off to a not-yet-migrated page returns to the legacy switch
      //    (the migrated -> ?page= transition that used to double-navigate).
      await page.locator("a", { hasText: "Virtual Keys" }).click();
      await expect(page).toHaveURL(new RegExp(`${ROOT}/ui/\\?page=api-keys`));
      await dismissFeedbackPopup(page);
      expect(pageErrors, `page errors after leaving ${ROOT}/ui/${segment}`).toEqual([]);
      await watch(page);
    });
  }
});
