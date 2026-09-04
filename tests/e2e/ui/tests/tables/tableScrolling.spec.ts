import { test, expect, type APIRequestContext, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { CHAT_MODEL_A, masterKey, sendChatCompletion, waitForSpendLog } from "../../helpers/traffic";

/**
 * LIT-4738 table scrolling. On the paginated pages the app shell <main> is the only page scroller
 * and must never overflow: rows scroll inside the table body under a header that stays put, and the
 * pagination footer sits at the bottom of the page instead of below the fold or inside a clipped
 * box. Pages that keep plain page scrolling must never paint rows past a fixed-height ancestor.
 * The viewport is pinned so "more rows than fit" means the same thing on every machine.
 */

const VIEWPORT = { width: 1280, height: 720 };
const SEED_ROWS = 40;
const LOG_ROWS = 20;
const BODY_SCROLL_PX = 500;
/** p-8 on Keys and Teams, p-6 on Logs: the footer may sit at most one page padding above the edge. */
const MAX_FOOTER_GAP_PX = 40;

const uniqueSuffix = (): string => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const adminHeaders = (): Record<string, string> => ({
  Authorization: `Bearer ${masterKey()}`,
});

/** Keys and Teams render a <main> of their own inside the app shell's, which comes first in document order. */
const pageScroller = (page: PlaywrightPage): Locator => page.locator("main").first();

/** Tabs keep every panel mounted, so a bare test id can match a hidden table; scope to the visible one. */
const visibleTestId = (page: PlaywrightPage, id: string): Locator => page.getByTestId(id).filter({ visible: true });

/** The page's data table; Model Hub also renders a plain links table above it, which this skips. */
const visibleDataTable = (page: PlaywrightPage): Locator => visibleTestId(page, "data-table-root").first();

const visibleRows = (page: PlaywrightPage): Locator => visibleDataTable(page).locator("tbody tr");

interface BoxMetrics {
  top: number;
  bottom: number;
  scrollHeight: number;
  clientHeight: number;
  scrollWidth: number;
  clientWidth: number;
}

const metrics = (locator: Locator): Promise<BoxMetrics> =>
  locator.evaluate((el) => {
    const rect = el.getBoundingClientRect();
    return {
      top: rect.top,
      bottom: rect.bottom,
      scrollHeight: el.scrollHeight,
      clientHeight: el.clientHeight,
      scrollWidth: el.scrollWidth,
      clientWidth: el.clientWidth,
    };
  });

async function postOk(
  request: APIRequestContext,
  path: string,
  data: Record<string, unknown>,
): Promise<Record<string, any>> {
  const res = await request.post(path, { headers: adminHeaders(), data });
  expect(res.ok(), `POST ${path} failed (${res.status()}): ${await res.text()}`).toBe(true);
  return (await res.json()) as Record<string, any>;
}

/** One request at a time: a burst of forty management calls starves the proxy's transaction pool. */
async function oneAtATime<T>(count: number, call: (index: number) => Promise<T>): Promise<T[]> {
  const results: T[] = [];
  for (let i = 0; i < count; i++) {
    results.push(await call(i));
  }
  return results;
}

async function expectRowsAtLeast(page: PlaywrightPage, count: number): Promise<void> {
  await expect.poll(() => visibleRows(page).count(), { timeout: 30_000 }).toBeGreaterThanOrEqual(count);
}

async function setRowsPerPage(page: PlaywrightPage, size: "25" | "50" | "100"): Promise<void> {
  await visibleTestId(page, "pagination-page-size").click();
  await page.getByRole("option", { name: size, exact: true }).click();
}

/**
 * The page scroller stays put, the table body is what scrolls, the header does not move while the
 * body scrolls, and the pagination footer sits at the bottom of the page.
 */
async function expectBodyIsTheOnlyScroller(page: PlaywrightPage): Promise<void> {
  const scroller = await metrics(pageScroller(page));
  const body = visibleTestId(page, "data-table-scroller");
  const bodyBefore = await metrics(body);
  const headBefore = await metrics(visibleTestId(page, "data-table-head"));
  const footer = await metrics(visibleDataTable(page));

  expect(scroller.scrollHeight, "page scroller must not overflow vertically").toBe(scroller.clientHeight);
  expect(scroller.scrollWidth, "page scroller must not overflow horizontally").toBe(scroller.clientWidth);
  expect(bodyBefore.scrollHeight, "table body must be the element that scrolls").toBeGreaterThan(
    bodyBefore.clientHeight,
  );
  expect(footer.bottom, "pagination footer must be inside the page").toBeLessThanOrEqual(scroller.bottom);
  expect(scroller.bottom - footer.bottom, "pagination footer must sit at the bottom of the page").toBeLessThanOrEqual(
    MAX_FOOTER_GAP_PX,
  );

  await body.evaluate((el, px) => {
    el.scrollTop = px;
  }, BODY_SCROLL_PX);
  await expect.poll(() => body.evaluate((el) => el.scrollTop)).toBeGreaterThan(0);
  const headAfter = await metrics(visibleTestId(page, "data-table-head"));
  expect(Math.round(headAfter.top), "header must stay put while the body scrolls").toBe(Math.round(headBefore.top));
}

/**
 * Every row must sit inside each ancestor up to the nearest one that really scrolls vertically; a
 * fixed-height box that neither grows nor scrolls lets rows paint past its bottom edge.
 */
const rowsPaintingPastAnAncestor = (page: PlaywrightPage): Promise<string[]> =>
  visibleDataTable(page)
    .locator("table")
    .evaluate((table) => {
      const scrollsVertically = (el: Element): boolean =>
        /auto|scroll/.test(getComputedStyle(el).overflowY) && el.scrollHeight > el.clientHeight + 1;
      const describe = (el: Element): string =>
        `<${el.tagName.toLowerCase()} class="${el.getAttribute("class") ?? ""}">`;
      return Array.from(table.querySelectorAll("tbody tr")).flatMap((row, index) => {
        const rowBottom = row.getBoundingClientRect().bottom;
        const spills: string[] = [];
        for (let el = row.parentElement; el && el !== document.body && !scrollsVertically(el); el = el.parentElement) {
          const bottom = el.getBoundingClientRect().bottom;
          if (rowBottom > bottom + 1) {
            spills.push(
              `row ${index} bottom ${Math.round(rowBottom)} past ${describe(el)} bottom ${Math.round(bottom)}`,
            );
          }
        }
        return spills;
      });
    });

type Cleanup = (request: APIRequestContext) => Promise<unknown>;
const cleanups: Cleanup[] = [];

test.describe("Admin tables scroll inside the page", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH, viewport: VIEWPORT });

  test.afterEach(async ({ request }) => {
    for (const cleanup of cleanups.splice(0)) {
      // Teardown must never turn a passing test red or mask a real failure.
      await cleanup(request).catch(() => {});
    }
  });

  test("Virtual Keys: rows scroll under a sticky header and the page itself never scrolls", async ({
    page,
    request,
  }) => {
    const suffix = uniqueSuffix();
    const created = await oneAtATime(SEED_ROWS, (i) =>
      postOk(request, "/key/generate", { key_alias: `e2e-scroll-key-${suffix}-${i}` }),
    );
    cleanups.push((r) => r.post("/key/delete", { headers: adminHeaders(), data: { keys: created.map((k) => k.key) } }));

    await navigateToPage(page, Page.ApiKeys);
    await expectRowsAtLeast(page, SEED_ROWS);
    await expectBodyIsTheOnlyScroller(page);
  });

  test("Teams: rows scroll under a sticky header and the page itself never scrolls", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const created = await oneAtATime(SEED_ROWS, (i) =>
      postOk(request, "/team/new", { team_alias: `e2e-scroll-team-${suffix}-${i}` }),
    );
    cleanups.push((r) =>
      r.post("/team/delete", { headers: adminHeaders(), data: { team_ids: created.map((t) => t.team_id) } }),
    );

    await navigateToPage(page, Page.Teams);
    await expectRowsAtLeast(page, SEED_ROWS);
    await expectBodyIsTheOnlyScroller(page);
  });

  test("Request Logs: rows scroll under a sticky header and the page itself never scrolls", async ({
    page,
    request,
  }) => {
    const suffix = uniqueSuffix();
    const ids = await oneAtATime(LOG_ROWS, (i) =>
      sendChatCompletion(request, { model: CHAT_MODEL_A, prompt: `scroll ${suffix} ${i}` }),
    );
    await waitForSpendLog(request, ids[ids.length - 1]);

    await navigateToPage(page, Page.Logs);
    await expect(visibleTestId(page, "datatable-search")).toBeVisible({ timeout: 20_000 });
    await setRowsPerPage(page, "25");
    await expectRowsAtLeast(page, LOG_ROWS);
    await expectBodyIsTheOnlyScroller(page);
  });

  test("Tags: no row paints past the box it lives in", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const names = Array.from({ length: SEED_ROWS }, (_, i) => `e2e-scroll-tag-${suffix}-${i}`);
    await oneAtATime(SEED_ROWS, (i) => postOk(request, "/tag/new", { name: names[i], description: "LIT-4738 scroll" }));
    cleanups.push((r) =>
      oneAtATime(SEED_ROWS, (i) => r.post("/tag/delete", { headers: adminHeaders(), data: { name: names[i] } })),
    );

    await navigateToPage(page, Page.TagManagement);
    await expectRowsAtLeast(page, SEED_ROWS);
    expect(await rowsPaintingPastAnAncestor(page)).toEqual([]);
  });

  test("Model Hub: no row paints past the box it lives in", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const created = await oneAtATime(SEED_ROWS, (i) =>
      postOk(request, "/model/new", {
        model_name: `e2e-scroll-model-${suffix}-${i}`,
        litellm_params: {
          model: "openai/fake-gpt-4",
          api_base: `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`,
          api_key: "fake-key",
        },
      }),
    );
    cleanups.push((r) =>
      oneAtATime(SEED_ROWS, (i) =>
        r.post("/model/delete", { headers: adminHeaders(), data: { id: created[i].model_info.id } }),
      ),
    );

    await navigateToPage(page, Page.ModelHubTable);
    await expectRowsAtLeast(page, SEED_ROWS);
    expect(await rowsPaintingPastAnAncestor(page)).toEqual([]);
  });
});
