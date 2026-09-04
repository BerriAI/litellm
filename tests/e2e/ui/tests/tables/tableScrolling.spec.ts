import { test, expect, type APIRequestContext, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { CHAT_MODEL_A, masterKey, sendChatCompletion, waitForSpendLog } from "../../helpers/traffic";

const VIEWPORT = { width: 1280, height: 720 };
const SEED_ROWS = 40;
const LOG_ROWS = 20;
const BODY_SCROLL_PX = 500;
const MAX_FOOTER_GAP_PX = 40;

interface GeneratedKey {
  key: string;
}

interface CreatedTeam {
  team_id: string;
}

interface CreatedModel {
  model_info: { id: string };
}

interface BoxMetrics {
  top: number;
  bottom: number;
  scrollHeight: number;
  clientHeight: number;
  scrollWidth: number;
  clientWidth: number;
}

const uniqueSuffix = (): string => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

const adminHeaders = (): Record<string, string> => ({ Authorization: `Bearer ${masterKey()}` });

const appShellMain = (page: PlaywrightPage): Locator => page.locator("main").first();

const visibleTestId = (page: PlaywrightPage, id: string): Locator => page.getByTestId(id).filter({ visible: true });

const visibleDataTable = (page: PlaywrightPage): Locator => visibleTestId(page, "data-table-root").first();

const visibleRows = (page: PlaywrightPage): Locator => visibleDataTable(page).locator("tbody tr");

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

async function postOk<T>(request: APIRequestContext, path: string, data: Record<string, unknown>): Promise<T> {
  const res = await request.post(path, { headers: adminHeaders(), data });
  expect(res.ok(), `POST ${path} failed (${res.status()}): ${await res.text()}`).toBe(true);
  return (await res.json()) as T;
}

const oneAtATime = <T>(count: number, call: (index: number) => Promise<T>): Promise<readonly T[]> =>
  Array.from({ length: count }, (_, i) => i).reduce<Promise<readonly T[]>>(
    async (previous, i) => [...(await previous), await call(i)],
    Promise.resolve([]),
  );

async function expectRowsAtLeast(page: PlaywrightPage, count: number): Promise<void> {
  await expect.poll(() => visibleRows(page).count(), { timeout: 30_000 }).toBeGreaterThanOrEqual(count);
}

async function setRowsPerPage(page: PlaywrightPage, size: "25" | "50" | "100"): Promise<void> {
  await visibleTestId(page, "pagination-page-size").click();
  await page.getByRole("option", { name: size, exact: true }).click();
}

async function expectBodyIsTheOnlyScroller(page: PlaywrightPage): Promise<void> {
  const scroller = await metrics(appShellMain(page));
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

const rowsPaintingPastAnAncestor = (page: PlaywrightPage): Promise<string[]> =>
  visibleDataTable(page)
    .locator("table")
    .evaluate((table) => {
      const scrollsVertically = (el: Element): boolean =>
        /auto|scroll/.test(getComputedStyle(el).overflowY) && el.scrollHeight > el.clientHeight + 1;
      const boxesUpToTheScroller = (el: Element | null): Element[] =>
        el === null || el === document.body || scrollsVertically(el)
          ? []
          : [el, ...boxesUpToTheScroller(el.parentElement)];
      const describe = (el: Element): string =>
        `<${el.tagName.toLowerCase()} class="${el.getAttribute("class") ?? ""}">`;
      return Array.from(table.querySelectorAll("tbody tr")).flatMap((row, index) => {
        const rowBottom = row.getBoundingClientRect().bottom;
        return boxesUpToTheScroller(row.parentElement)
          .filter((box) => rowBottom > box.getBoundingClientRect().bottom + 1)
          .map(
            (box) =>
              `row ${index} bottom ${Math.round(rowBottom)} past ${describe(box)} bottom ${Math.round(box.getBoundingClientRect().bottom)}`,
          );
      });
    });

test.describe("Admin tables scroll inside the page", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH, viewport: VIEWPORT });

  test("Virtual Keys: rows scroll under a sticky header and the page itself never scrolls", async ({
    page,
    request,
  }) => {
    const suffix = uniqueSuffix();
    const keys = await oneAtATime(SEED_ROWS, (i) =>
      postOk<GeneratedKey>(request, "/key/generate", { key_alias: `e2e-scroll-key-${suffix}-${i}` }),
    );
    try {
      await navigateToPage(page, Page.ApiKeys);
      await expectRowsAtLeast(page, SEED_ROWS);
      await expectBodyIsTheOnlyScroller(page);
    } finally {
      await request.post("/key/delete", { headers: adminHeaders(), data: { keys: keys.map((k) => k.key) } });
    }
  });

  test("Teams: rows scroll under a sticky header and the page itself never scrolls", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const teams = await oneAtATime(SEED_ROWS, (i) =>
      postOk<CreatedTeam>(request, "/team/new", { team_alias: `e2e-scroll-team-${suffix}-${i}` }),
    );
    try {
      await navigateToPage(page, Page.Teams);
      await expectRowsAtLeast(page, SEED_ROWS);
      await expectBodyIsTheOnlyScroller(page);
    } finally {
      await request.post("/team/delete", { headers: adminHeaders(), data: { team_ids: teams.map((t) => t.team_id) } });
    }
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
    await oneAtATime(SEED_ROWS, (i) =>
      postOk<unknown>(request, "/tag/new", { name: names[i], description: "LIT-4738 scroll" }),
    );
    try {
      await navigateToPage(page, Page.TagManagement);
      await expectRowsAtLeast(page, SEED_ROWS);
      expect(await rowsPaintingPastAnAncestor(page)).toEqual([]);
    } finally {
      await oneAtATime(SEED_ROWS, (i) =>
        request.post("/tag/delete", { headers: adminHeaders(), data: { name: names[i] } }),
      );
    }
  });

  test("Model Hub: no row paints past the box it lives in", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const models = await oneAtATime(SEED_ROWS, (i) =>
      postOk<CreatedModel>(request, "/model/new", {
        model_name: `e2e-scroll-model-${suffix}-${i}`,
        litellm_params: {
          model: "openai/fake-gpt-4",
          api_base: `http://127.0.0.1:${process.env.MOCK_LLM_PORT ?? "8090"}/v1`,
          api_key: "fake-key",
        },
      }),
    );
    try {
      await navigateToPage(page, Page.ModelHubTable);
      await expectRowsAtLeast(page, SEED_ROWS);
      expect(await rowsPaintingPastAnAncestor(page)).toEqual([]);
    } finally {
      await oneAtATime(SEED_ROWS, (i) =>
        request.post("/model/delete", { headers: adminHeaders(), data: { id: models[i].model_info.id } }),
      );
    }
  });
});
