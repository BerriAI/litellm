import { test, expect, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { CHAT_MODEL_A, MOCK_RESPONSE_TEXT, sendChatCompletion, waitForSpendLog } from "../../helpers/traffic";

/**
 * Logs page manual-QA coverage: a request the proxy actually served shows up in
 * the table, its detail drawer expands to the real request and response bodies,
 * both can be copied, and the End User filter narrows the table to it.
 *
 * Every assertion is anchored to traffic this spec generates itself (unique
 * prompt + unique end user per run), so it neither depends on seeded spend rows
 * nor collides with the other specs' traffic when the suite runs in parallel.
 */

const uniqueSuffix = (): string => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

/**
 * The Input/Output cards in the drawer are built from SectionHeader, whose
 * label is an antd Text span nested icon-div > flex-row > header-root. Walking
 * up from the label is the only stable handle: the header has no role, test id
 * or class of its own, and its copy button is icon-only (its "Copy" tooltip
 * only exists while hovered, so it has no accessible name to query by).
 */
const sectionHeader = (drawer: Locator, label: "Input" | "Output"): Locator =>
  drawer.getByText(label, { exact: true }).locator("xpath=../../..");

/**
 * The Logs page mounts every tab (Request Logs, Audit Logs, Deleted Keys,
 * Deleted Teams), so the DOM holds four tables and four data-table toolbars at
 * once and only the active tab's are visible. Unscoped `table tbody tr` counts
 * rows from all four; every locator here goes through the visible one.
 */
const requestLogsRows = (page: PlaywrightPage): Locator =>
  page.locator("table").filter({ visible: true }).first().locator("tbody tr");

const visibleTestId = (page: PlaywrightPage, id: string): Locator => page.getByTestId(id).filter({ visible: true });

/** Open the Logs page and filter the table down to a single request id. */
async function openLogsForRequest(page: PlaywrightPage, requestId: string): Promise<Locator> {
  await navigateToPage(page, Page.Logs);
  await dismissFeedbackPopup(page);

  const search = visibleTestId(page, "datatable-search");
  await expect(search).toBeVisible({ timeout: 20_000 });
  await search.fill(requestId);

  const row = requestLogsRows(page).filter({ hasText: requestId });
  await expect(row, `no logs row for request ${requestId}`).toHaveCount(1, {
    timeout: 30_000,
  });
  return row;
}

test.describe("Logs page", () => {
  test.use({
    storageState: ADMIN_STORAGE_PATH,
    // The drawer's copy buttons go through navigator.clipboard; without these
    // the writes reject and the success toast never fires.
    permissions: ["clipboard-read", "clipboard-write"],
  });

  test("a served request expands to its request and response, and both copy", async ({ page, request }) => {
    const prompt = `logs-detail-prompt-${uniqueSuffix()}`;
    const requestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt,
    });
    await waitForSpendLog(request, requestId);

    const row = await openLogsForRequest(page, requestId);

    // Expand: clicking the row opens the detail drawer for that request.
    await row.click();
    const drawer = page.locator(".ant-drawer-content").first();
    await expect(drawer).toBeVisible({ timeout: 20_000 });
    await expect(drawer.getByText("Request & Response")).toBeVisible({
      timeout: 20_000,
    });

    // The prompt we sent and the mock server's reply are both rendered.
    await expect(drawer.getByText(prompt, { exact: false })).toBeVisible({
      timeout: 20_000,
    });
    await expect(drawer.getByText(MOCK_RESPONSE_TEXT, { exact: false }).first()).toBeVisible({ timeout: 20_000 });

    // Copy request: the Input card's copy button puts the prompt on the clipboard.
    await sectionHeader(drawer, "Input").getByRole("button").click();
    await expect(page.getByText("Input copied")).toBeVisible({
      timeout: 10_000,
    });
    expect(await page.evaluate(() => navigator.clipboard.readText())).toContain(prompt);

    // Copy response: the Output card's copy button puts the completion on it.
    await sectionHeader(drawer, "Output").getByRole("button").click();
    await expect(page.getByText("Output copied")).toBeVisible({
      timeout: 10_000,
    });
    expect(await page.evaluate(() => navigator.clipboard.readText())).toContain(MOCK_RESPONSE_TEXT);
  });

  test("the Input card collapses and expands", async ({ page, request }) => {
    const prompt = `logs-collapse-prompt-${uniqueSuffix()}`;
    const requestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt,
    });
    await waitForSpendLog(request, requestId);

    const row = await openLogsForRequest(page, requestId);
    await row.click();

    const drawer = page.locator(".ant-drawer-content").first();
    await expect(drawer.getByText("Request & Response")).toBeVisible({
      timeout: 20_000,
    });

    // SectionHeader renders an up-arrow while expanded and a down-arrow while
    // collapsed. The body is the header's next sibling and collapses via
    // `max-height: 0; overflow: hidden`, which zeroes its own bounding box —
    // so the wrapper reads as hidden even though the prompt text node inside
    // it does not (its box keeps its size, it is merely clipped by the parent).
    const header = sectionHeader(drawer, "Input");
    const body = header.locator("xpath=following-sibling::div[1]");
    await expect(header.locator(".anticon-up")).toBeVisible();
    await expect(body).toBeVisible();

    await header.click();
    await expect(header.locator(".anticon-down")).toBeVisible({
      timeout: 10_000,
    });
    await expect(body).toBeHidden({ timeout: 10_000 });

    await header.click();
    await expect(header.locator(".anticon-up")).toBeVisible({
      timeout: 10_000,
    });
    await expect(body).toBeVisible({ timeout: 10_000 });
    await expect(drawer.getByText(prompt, { exact: false })).toBeVisible({
      timeout: 10_000,
    });
  });

  test("the JSON view exposes Request and Response tabs", async ({ page, request }) => {
    const prompt = `logs-json-prompt-${uniqueSuffix()}`;
    const requestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt,
    });
    await waitForSpendLog(request, requestId);

    const row = await openLogsForRequest(page, requestId);
    await row.click();

    const drawer = page.locator(".ant-drawer-content").first();
    await expect(drawer.getByText("Request & Response")).toBeVisible({
      timeout: 20_000,
    });

    // antd Radio.Button hides the <input> under its <label>, which intercepts
    // the pointer event — click the label, not the radio.
    await drawer.locator("label.ant-radio-button-wrapper").filter({ hasText: "JSON" }).click();

    const requestTab = drawer.getByRole("tab", { name: "Request" });
    await expect(requestTab).toBeVisible({ timeout: 10_000 });
    await requestTab.click();
    await expect(drawer.getByText(prompt, { exact: false }).first()).toBeVisible({ timeout: 10_000 });

    await drawer.getByRole("tab", { name: "Response" }).click();
    await expect(drawer.getByText(MOCK_RESPONSE_TEXT, { exact: false }).first()).toBeVisible({ timeout: 10_000 });
  });

  test("the End User filter narrows the table to that customer", async ({ page, request }) => {
    const endUser = `logs-end-user-${uniqueSuffix()}`;
    const minePrompt = `logs-filter-mine-${uniqueSuffix()}`;
    const otherPrompt = `logs-filter-other-${uniqueSuffix()}`;

    const mineId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: minePrompt,
      endUser,
    });
    const otherId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: otherPrompt,
    });
    await waitForSpendLog(request, mineId);
    await waitForSpendLog(request, otherId);

    await navigateToPage(page, Page.Logs);
    await dismissFeedbackPopup(page);

    // Both requests are in the unfiltered table.
    await expect(requestLogsRows(page).filter({ hasText: mineId })).toHaveCount(1, { timeout: 30_000 });
    await expect(requestLogsRows(page).filter({ hasText: otherId })).toHaveCount(1, { timeout: 30_000 });

    await visibleTestId(page, "datatable-filters-trigger").click();
    const filters = page.getByRole("dialog").filter({ hasText: "Narrow down request logs" });
    await expect(filters).toBeVisible({ timeout: 10_000 });

    const endUserInput = filters.getByPlaceholder("Search an end user");
    await endUserInput.click();
    await endUserInput.fill(endUser);
    // The combobox popup is portaled to the body, so it is outside the filter
    // dialog's subtree — scope the option lookup to the page, not the dialog.
    await page.getByRole("option", { name: endUser, exact: true }).click({ timeout: 30_000 });
    await filters.getByRole("button", { name: "Apply Filters" }).click();

    // Only the request tagged with this end user survives the filter.
    await expect(requestLogsRows(page).filter({ hasText: otherId })).toHaveCount(0, { timeout: 30_000 });
    await expect(requestLogsRows(page).filter({ hasText: mineId })).toHaveCount(1);
    await expect(requestLogsRows(page)).toHaveCount(1);
  });
});
