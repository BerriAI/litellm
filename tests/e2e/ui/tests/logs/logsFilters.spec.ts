import { test, expect, type APIRequestContext, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import {
  CHAT_MODEL_A,
  CHAT_MODEL_B,
  createVirtualKey,
  sendChatCompletion,
  waitForSpendLog,
} from "../../helpers/traffic";

/**
 * Every test mints its own key and asserts against request ids it generated, so a filter that
 * quietly does nothing shows up as the other key's row still being on screen, and concurrent
 * specs' traffic cannot decide the outcome.
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

/** Picks a value in one of the drawer's searchable comboboxes and applies the filter. */
async function applyComboboxFilter(
  page: PlaywrightPage,
  drawer: Locator,
  comboboxLabel: string,
  value: string,
): Promise<void> {
  await drawer.getByRole("combobox", { name: comboboxLabel }).click();
  await page.keyboard.type(value);
  await page.getByRole("option", { name: value, exact: true }).first().click();
  await drawer.getByRole("button", { name: "Apply Filters" }).click();
  await expect(drawer).not.toBeVisible({ timeout: 10_000 });
}

/** A request the key is not entitled to make, so the proxy refuses it and logs the refusal. */
async function sendDeniedCompletion(request: APIRequestContext, apiKey: string): Promise<void> {
  const res = await request.post("/v1/chat/completions", {
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    data: { model: CHAT_MODEL_B, messages: [{ role: "user", content: "denied" }] },
  });
  expect(res.status(), "a model outside the key's allow-list is refused").toBe(403);
}

test.describe("Logs page filters", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("the Key Alias filter narrows the table to that key's requests", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const mine = await createVirtualKey(request, { key_alias: `e2e-logs-mine-${suffix}` });
    const theirs = await createVirtualKey(request, { key_alias: `e2e-logs-theirs-${suffix}` });

    const myRequestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: `logs-filter-mine-${suffix}`,
      apiKey: mine.key,
    });
    const theirRequestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: `logs-filter-theirs-${suffix}`,
      apiKey: theirs.key,
    });
    await waitForSpendLog(request, myRequestId);
    await waitForSpendLog(request, theirRequestId);

    await openLogs(page);
    const drawer = await openFilterDrawer(page);
    await applyComboboxFilter(page, drawer, "Search a key alias", mine.alias!);

    await expect(requestLogsRows(page).filter({ hasText: myRequestId })).toHaveCount(1, { timeout: 30_000 });
    // The filter is only doing its job if the other key's request is gone, not merely if ours is present.
    await expect(requestLogsRows(page).filter({ hasText: theirRequestId })).toHaveCount(0, { timeout: 10_000 });
  });

  test("the Status filter narrows the table to the refused request", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const alias = `e2e-logs-status-${suffix}`;
    const scoped = await createVirtualKey(request, { key_alias: alias, models: [CHAT_MODEL_A] });

    const servedRequestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: `logs-filter-served-${suffix}`,
      apiKey: scoped.key,
    });
    await sendDeniedCompletion(request, scoped.key);
    await waitForSpendLog(request, servedRequestId);

    await openLogs(page);
    const drawer = await openFilterDrawer(page);
    await drawer.getByRole("combobox", { name: "Search a key alias" }).click();
    await page.keyboard.type(alias);
    await page.getByRole("option", { name: alias, exact: true }).first().click();
    // The Status field labels its group, not the trigger, so it is addressed by the value it shows.
    await drawer.getByRole("combobox").filter({ hasText: "All Statuses" }).click();
    await page.getByRole("option", { name: "Failure", exact: true }).click();
    await drawer.getByRole("button", { name: "Apply Filters" }).click();
    await expect(drawer).not.toBeVisible({ timeout: 10_000 });

    // Both requests were made by this key, so a Status filter that does nothing leaves the served one on screen.
    await expect(requestLogsRows(page)).toHaveCount(1, { timeout: 30_000 });
    await expect(requestLogsRows(page)).toContainText("Failure");
    await expect(requestLogsRows(page).filter({ hasText: servedRequestId })).toHaveCount(0);
  });

  test("Reset Filters brings back the rows a filter hid", async ({ page, request }) => {
    const suffix = uniqueSuffix();
    const mine = await createVirtualKey(request, { key_alias: `e2e-logs-reset-mine-${suffix}` });
    const theirs = await createVirtualKey(request, { key_alias: `e2e-logs-reset-theirs-${suffix}` });

    const myRequestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: `logs-reset-mine-${suffix}`,
      apiKey: mine.key,
    });
    const theirRequestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: `logs-reset-theirs-${suffix}`,
      apiKey: theirs.key,
    });
    await waitForSpendLog(request, myRequestId);
    await waitForSpendLog(request, theirRequestId);

    await openLogs(page);
    const drawer = await openFilterDrawer(page);
    await applyComboboxFilter(page, drawer, "Search a key alias", mine.alias!);
    await expect(requestLogsRows(page).filter({ hasText: theirRequestId })).toHaveCount(0, { timeout: 30_000 });

    // A filter you cannot clear is a page that looks empty forever, which is how it reads to a user.
    await page.getByRole("button", { name: "Reset Filters" }).filter({ visible: true }).click();

    await expect(requestLogsRows(page).filter({ hasText: theirRequestId })).toHaveCount(1, { timeout: 30_000 });
    await expect(requestLogsRows(page).filter({ hasText: myRequestId })).toHaveCount(1, { timeout: 10_000 });
  });
});
