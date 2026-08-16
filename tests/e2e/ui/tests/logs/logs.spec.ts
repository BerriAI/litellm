import { test, expect, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import { CHAT_MODEL_A, MOCK_RESPONSE_TEXT, sendChatCompletion, waitForSpendLog } from "../../helpers/traffic";

/**
 * Anchored to traffic this spec generates itself, with a unique prompt and end user per run, so it
 * neither depends on seeded spend rows nor collides with other specs under parallelism.
 */

const uniqueSuffix = (): string => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

/**
 * Walking up from the label is the only stable handle: the header carries no role, test id or class,
 * and its copy button is icon-only with a hover-only tooltip.
 */
const sectionHeader = (drawer: Locator, label: "Input" | "Output"): Locator =>
  drawer.getByText(label, { exact: true }).locator("xpath=../../..");

/** Every tab stays mounted, so the DOM holds four tables at once; scope to the visible one. */
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
    // The copy buttons go through navigator.clipboard, which rejects without these.
    permissions: ["clipboard-read", "clipboard-write"],
  });

  test("a served request expands to its request and response", async ({ page, request }) => {
    const prompt = `logs-detail-prompt-${uniqueSuffix()}`;
    const requestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt,
    });
    await waitForSpendLog(request, requestId);

    const row = await openLogsForRequest(page, requestId);

    // Expand: clicking the row opens the detail drawer for that request.
    await row.click();
    const drawer = page.getByRole("dialog").first();
    await expect(drawer).toBeVisible({ timeout: 20_000 });
    await expect(drawer.getByText("Request & Response")).toBeVisible({
      timeout: 20_000,
    });

    // The prompt we sent and the mock server's reply are both rendered.
    await expect(drawer.getByText(prompt, { exact: false })).toBeVisible({
      timeout: 20_000,
    });
    await expect(drawer.getByText(MOCK_RESPONSE_TEXT, { exact: false }).first()).toBeVisible({ timeout: 20_000 });
  });

  // Split out because only the copy path needs a secure context; folding it in would
  // take the drawer-rendering coverage down with it.
  test("the drawer copies the request and the response to the clipboard", async ({ page, request }) => {
    // `navigator.clipboard` is undefined outside a secure context, and handleCopy calls
    // writeText unguarded, so on plain HTTP served from a hostname the click throws and no
    // toast renders. Skipped rather than weakened so the product gap stays visible.
    await page.goto("/ui");
    const isSecure = await page.evaluate(() => window.isSecureContext);
    test.skip(!isSecure, "origin is not a secure context, so navigator.clipboard is unavailable");

    const prompt = `logs-copy-prompt-${uniqueSuffix()}`;
    const requestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt,
    });
    await waitForSpendLog(request, requestId);

    const row = await openLogsForRequest(page, requestId);
    await row.click();
    const drawer = page.getByRole("dialog").first();
    await expect(drawer).toBeVisible({ timeout: 20_000 });

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

    const drawer = page.getByRole("dialog").first();
    await expect(drawer.getByText("Request & Response")).toBeVisible({
      timeout: 20_000,
    });

    // The body collapses via `max-height: 0; overflow: hidden`, which zeroes its own bounding
    // box, so the wrapper reads as hidden while the clipped text node inside it does not.
    const header = sectionHeader(drawer, "Input");
    const body = header.locator("xpath=following-sibling::div[1]");
    await expect(header.locator(".lucide-chevron-up")).toBeVisible();
    await expect(body).toBeVisible();

    await header.click();
    await expect(header.locator(".lucide-chevron-down")).toBeVisible({
      timeout: 10_000,
    });
    await expect(body).toBeHidden({ timeout: 10_000 });

    await header.click();
    await expect(header.locator(".lucide-chevron-up")).toBeVisible({
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

    const drawer = page.getByRole("dialog").first();
    await expect(drawer.getByText("Request & Response")).toBeVisible({
      timeout: 20_000,
    });

    await drawer.getByRole("tab", { name: "JSON" }).click();

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
