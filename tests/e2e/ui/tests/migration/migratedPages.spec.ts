import { test, expect, type Page } from "@playwright/test";
import { MIGRATED_E2E_SEGMENTS } from "../../fixtures/migratedPages";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { dismissFeedbackPopup } from "../../helpers/navigation";

/**
 * App Router migration smoke as a user journey: start where the proxy lands you,
 * click a migrated page in the sidebar, confirm it routed and rendered, reload it
 * (the check a wrong server_root_path breaks), bounce to a legacy page and back,
 * and, once two pages are migrated, navigate directly between two migrated pages.
 *
 * Driven by MIGRATED_E2E_SEGMENTS, so it grows as pages are migrated. Set
 * SERVER_ROOT_PATH (e.g. "/litellm") to exercise the non-root mount; leave it
 * unset for the default mount. Boot the proxy with the matching value first.
 */
const ROOT = process.env.SERVER_ROOT_PATH ?? "";

const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const pathRe = (segment: string) => new RegExp(`${esc(ROOT)}/ui/${esc(segment)}/?($|\\?)`);
// Scope nav lookups to the sidebar (a `complementary` landmark). The top bar
// now renders a breadcrumb whose current-page item is also a "Virtual Keys"
// link, so an unscoped locator would match two elements.
const sidebar = (page: Page) => page.getByRole("complementary");
const virtualKeysLink = (page: Page) => sidebar(page).getByRole("link", { name: "Virtual Keys", exact: true });

/**
 * Segments whose journey currently fails on ambient environment rather than on
 * the migration this spec exists to check.
 *
 * These run green locally and in CircleCI, where the proxy owns its own schema
 * on a fresh database. They fail against a deployed stack whose schema is
 * created by a separate migrations Job, because the page's data calls 500 and
 * `pageerror` collects the stack trace.
 *
 * Keyed by segment so the rest of the matrix keeps running — this is not a
 * blanket disable of the file.
 */
const AMBIENT_ENV_SKIPS: Record<string, string> = {
  "old-usage":
    'GET /spend/tags 500s with relation "DailyTagSpend" does not exist. The spend views are ' +
    "created by a fire-and-forget task at proxy startup (proxy_server.py: check_view_exists), " +
    "so on a deployed stack they can be absent with no error surfaced anywhere. Not a migration " +
    "regression: the page routes and renders, only its data calls fail.",
};

/** The dashboard shell is present (sidebar rendered); page didn't 404 / crash. */
async function expectRendered(page: Page) {
  await expect(virtualKeysLink(page)).toBeVisible({ timeout: 20_000 });
}

/**
 * Click a migrated page's sidebar link. Migrated items render as <a href=".../ui/<segment>">;
 * nested ones live under collapsible groups whose children only render while the
 * group is open, so expand collapsed groups until the link is clickable.
 */
async function clickSidebar(page: Page, segment: string) {
  const link = sidebar(page).locator(`a[href$="/ui/${segment}"]`).first();
  for (let i = 0; i < 8 && !(await link.isVisible().catch(() => false)); i++) {
    // A collapsed group is a menu item with a group-toggle button but no
    // rendered submenu yet; clicking the toggle expands it.
    const collapsedGroup = sidebar(page)
      .locator(
        '[data-slot="sidebar-menu-item"]:has(> [data-slot="sidebar-menu-button"]):not(:has(> [data-slot="sidebar-menu-sub"])) > [data-slot="sidebar-menu-button"]',
      )
      .first();
    if (!(await collapsedGroup.isVisible().catch(() => false))) break;
    await collapsedGroup.click();
    await page.waitForTimeout(250);
  }
  await link.click();
}

test.use({ storageState: ADMIN_STORAGE_PATH });

test.describe("App Router migrated pages", () => {
  for (const segment of MIGRATED_E2E_SEGMENTS) {
    test(`${segment}: sidebar nav, reload, and round-trip via the api-keys landing`, async ({ page }) => {
      test.skip(segment in AMBIENT_ENV_SKIPS, AMBIENT_ENV_SKIPS[segment]);
      const pageErrors: string[] = [];
      page.on("pageerror", (e) => pageErrors.push(String(e)));

      // 1. Start where the proxy lands us.
      await page.goto(`${ROOT}/ui/`);
      await dismissFeedbackPopup(page);
      await expectRendered(page);

      // 2. Click the migrated page in the sidebar -> path route + rendered.
      await clickSidebar(page, segment);
      await expect(page).toHaveURL(pathRe(segment));
      await expectRendered(page);
      // 3. Reload the path route directly; a wrong server_root_path 404s here.
      await page.reload();
      await dismissFeedbackPopup(page);
      await expect(page).toHaveURL(pathRe(segment));
      await expectRendered(page);
      // 4. Click the Virtual Keys sidebar link to the api-keys landing (now a path route), then back.
      await virtualKeysLink(page).click();
      await expect(page).toHaveURL(pathRe("api-keys"));
      await dismissFeedbackPopup(page);
      await expectRendered(page);
      // 5. Click back to the migrated page.
      await clickSidebar(page, segment);
      await expect(page).toHaveURL(pathRe(segment));
      await expectRendered(page);
      expect(pageErrors, `page errors during ${segment} journey`).toEqual([]);
    });
  }

  test("navigates directly between two migrated pages", async ({ page }) => {
    test.skip(MIGRATED_E2E_SEGMENTS.length < 2, "needs >= 2 migrated pages");
    const [first, second] = MIGRATED_E2E_SEGMENTS;
    const pageErrors: string[] = [];
    page.on("pageerror", (e) => pageErrors.push(String(e)));

    await page.goto(`${ROOT}/ui/`);
    await dismissFeedbackPopup(page);

    await clickSidebar(page, first);
    await expect(page).toHaveURL(pathRe(first));
    await expectRendered(page);
    await clickSidebar(page, second);
    await expect(page).toHaveURL(pathRe(second));
    await expectRendered(page);
    // Back to the first migrated page.
    await clickSidebar(page, first);
    await expect(page).toHaveURL(pathRe(first));
    await expectRendered(page);

    expect(pageErrors, "page errors during migrated -> migrated nav").toEqual([]);
  });
});
