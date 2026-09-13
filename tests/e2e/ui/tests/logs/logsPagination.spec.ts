import { test, expect, type APIRequestContext, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { CHAT_MODEL_A, createVirtualKey, sendChatCompletion, waitForSpendLog } from "../../helpers/traffic";

/**
 * Session-grouped pagination (#38060): a page of N rows must render exactly N session rows, a
 * session must never straddle pages, and two callers reusing one session id stay separate rows.
 * All traffic is generated per run behind a unique key alias or session id, so concurrent specs
 * cannot decide the outcome.
 */

const uniqueSuffix = (): string => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

/** Every tab stays mounted, so the DOM holds four tables at once; scope to the visible one. */
const requestLogsRows = (page: PlaywrightPage): Locator =>
  page.locator("table").filter({ visible: true }).first().locator("tbody tr");

const visibleTestId = (page: PlaywrightPage, id: string): Locator => page.getByTestId(id).filter({ visible: true });

async function openLogs(page: PlaywrightPage): Promise<void> {
  await navigateToPage(page, Page.Logs);
  await dismissFeedbackPopup(page);
  await expect(visibleTestId(page, "datatable-search")).toBeVisible({ timeout: 20_000 });
}

async function openFilterDrawer(page: PlaywrightPage): Promise<Locator> {
  await visibleTestId(page, "datatable-filters-trigger").click();
  const drawer = page.getByRole("dialog", { name: "Filters" });
  await expect(drawer).toBeVisible({ timeout: 10_000 });
  return drawer;
}

async function applyKeyAliasFilter(page: PlaywrightPage, drawer: Locator, alias: string): Promise<void> {
  await drawer.getByRole("combobox", { name: "Search a key alias" }).click();
  await page.keyboard.type(alias);
  await page.getByRole("option", { name: alias, exact: true }).first().click();
  await drawer.getByRole("button", { name: "Apply Filters" }).click();
  await expect(drawer).not.toBeVisible({ timeout: 10_000 });
}

async function setRowsPerPage(page: PlaywrightPage, size: "25" | "50" | "100"): Promise<void> {
  await visibleTestId(page, "pagination-page-size").click();
  await page.getByRole("option", { name: size, exact: true }).click();
}

test.describe("Logs page session-grouped pagination", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("a 25-row page renders exactly 25 session rows and no session straddles pages", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const alias = `e2e-logs-pgn-${suffix}`;
    const mine = await createVirtualKey(request, { key_alias: alias });

    const soloIds: string[] = [];
    for (let i = 0; i < 26; i++) {
      soloIds.push(
        await sendChatCompletion(request, {
          model: CHAT_MODEL_A,
          prompt: `logs-pgn-solo-${i}-${suffix}`,
          apiKey: mine.key,
        }),
      );
    }
    const sessionA = `sess-pgn-a-${suffix}`;
    const sessionB = `sess-pgn-b-${suffix}`;
    let lastSessionCallId = "";
    for (let i = 0; i < 7; i++) {
      lastSessionCallId = await sendChatCompletion(request, {
        model: CHAT_MODEL_A,
        prompt: `logs-pgn-a-${i}-${suffix}`,
        apiKey: mine.key,
        traceId: sessionA,
      });
    }
    for (let i = 0; i < 3; i++) {
      lastSessionCallId = await sendChatCompletion(request, {
        model: CHAT_MODEL_A,
        prompt: `logs-pgn-b-${i}-${suffix}`,
        apiKey: mine.key,
        traceId: sessionB,
      });
    }
    await waitForSpendLog(request, lastSessionCallId);
    await waitForSpendLog(request, soloIds[soloIds.length - 1]);

    // 36 calls in 28 session groups: 26 solos plus sessions of 7 and 3.
    await openLogs(page);
    const drawer = await openFilterDrawer(page);
    await applyKeyAliasFilter(page, drawer, alias);
    await setRowsPerPage(page, "25");

    await expect(visibleTestId(page, "pagination-range")).toHaveText("Showing 1-25 of 28", { timeout: 30_000 });
    await expect(requestLogsRows(page)).toHaveCount(25);
    // The sessions are the newest groups, so their single representative rows sit on page 1.
    await expect(requestLogsRows(page).filter({ hasText: sessionA })).toHaveCount(1);
    await expect(requestLogsRows(page).filter({ hasText: sessionA })).toContainText("7");
    await expect(requestLogsRows(page).filter({ hasText: sessionB })).toHaveCount(1);

    await visibleTestId(page, "pagination-next").click();

    await expect(visibleTestId(page, "pagination-range")).toHaveText("Showing 26-28 of 28", { timeout: 30_000 });
    await expect(requestLogsRows(page)).toHaveCount(3);
    await expect(requestLogsRows(page).filter({ hasText: sessionA })).toHaveCount(0);
    await expect(requestLogsRows(page).filter({ hasText: sessionB })).toHaveCount(0);
  });

  test("two keys reusing one session id stay separate rows", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const mine = await createVirtualKey(request, { key_alias: `e2e-logs-pgn-mine-${suffix}` });
    const theirs = await createVirtualKey(request, { key_alias: `e2e-logs-pgn-theirs-${suffix}` });
    const sharedSession = `sess-pgn-shared-${suffix}`;

    let lastId = "";
    for (let i = 0; i < 2; i++) {
      lastId = await sendChatCompletion(request, {
        model: CHAT_MODEL_A,
        prompt: `logs-pgn-shared-mine-${i}-${suffix}`,
        apiKey: mine.key,
        traceId: sharedSession,
      });
    }
    lastId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: `logs-pgn-shared-theirs-${suffix}`,
      apiKey: theirs.key,
      traceId: sharedSession,
    });
    await waitForSpendLog(request, lastId);

    await openLogs(page);
    const drawer = await openFilterDrawer(page);
    await drawer.getByPlaceholder("Enter session ID…").fill(sharedSession);
    await drawer.getByRole("button", { name: "Apply Filters" }).click();
    await expect(drawer).not.toBeVisible({ timeout: 10_000 });

    // One row per caller: reusing a session id must not merge two keys' activity into one row.
    await expect(requestLogsRows(page).filter({ hasText: sharedSession })).toHaveCount(2, { timeout: 30_000 });

    // And each row carries ITS key's totals: two calls badge the first key's row,
    // while the other key's single call renders as a plain LLM row.
    const mineRow = requestLogsRows(page).filter({ hasText: sharedSession }).filter({ hasText: mine.token });
    const theirsRow = requestLogsRows(page).filter({ hasText: sharedSession }).filter({ hasText: theirs.token });
    await expect(mineRow).toHaveCount(1);
    await expect(theirsRow).toHaveCount(1);
    await expect(mineRow.getByText("2", { exact: true })).toBeVisible();
    await expect(theirsRow.getByText("LLM", { exact: true })).toBeVisible();
  });
});
