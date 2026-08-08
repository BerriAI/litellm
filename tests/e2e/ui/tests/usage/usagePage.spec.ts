import { test, expect, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import {
  CHAT_MODEL_A,
  createVirtualKey,
  sendChatCompletion,
  waitForKeyInDailyActivity,
  waitForSpendLog,
} from "../../helpers/traffic";

/**
 * Usage page manual-QA coverage: traffic billed to a virtual key shows up in
 * Top Virtual Keys, the card switches between its table and chart renderings,
 * and clicking the key opens its key-info panel.
 *
 * Targets the current Usage page (/ui/usage). The legacy /ui/old-usage view
 * carries its own deprecation banner and is deliberately not covered here.
 */

/**
 * The Top Virtual Keys card. Its <Title> is a direct child of the card element,
 * so stepping up one level from the title is an exact handle — needed because
 * the page renders several other tables (Spend by Provider, Top Models) that an
 * unscoped table locator would pick up.
 */
const topKeysCard = (page: PlaywrightPage): Locator =>
  page.getByText("Top Virtual Keys", { exact: true }).locator("xpath=..");

async function openUsage(page: PlaywrightPage): Promise<Locator> {
  await navigateToPage(page, Page.NewUsage);
  await dismissFeedbackPopup(page);
  const card = topKeysCard(page);
  await expect(card).toBeVisible({ timeout: 30_000 });
  // Widen the leaderboard so the key under test is not cut off by the default
  // top-5 limit when the database already holds other keys. (antd Segmented
  // renders label-wrapped radios, not options.)
  await card.locator(".ant-segmented-item").filter({ hasText: /^50$/ }).click();
  return card;
}

test.describe("Usage page", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Top Virtual Keys lists a key that served traffic, toggles views, and opens key info", async ({
    page,
    request,
  }) => {
    const alias = `e2e-usage-key-${Date.now()}`;
    const { key, token } = await createVirtualKey(request, {
      key_alias: alias,
    });

    const requestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: `usage ping for ${alias}`,
      apiKey: key,
    });
    await waitForSpendLog(request, requestId);
    // Must land in the aggregate before the page mounts — it fetches once.
    await waitForKeyInDailyActivity(request, token);

    const card = await openUsage(page);

    // Table view (the default): the key is listed by its alias.
    const row = card.locator("tbody tr").filter({ hasText: alias });
    await expect(row, `${alias} missing from Top Virtual Keys`).toHaveCount(1, {
      timeout: 30_000,
    });

    // Chart view swaps the table out for the bar chart, and back.
    await card.getByText("Chart View", { exact: true }).click();
    await expect(card.locator("tbody tr")).toHaveCount(0, { timeout: 10_000 });
    await card.getByText("Table View", { exact: true }).click();
    await expect(row).toHaveCount(1, { timeout: 10_000 });

    // Clicking the Key ID cell fetches key info and opens the detail panel.
    // Assert on the panel's own controls, not on the alias: the alias is
    // already in the row behind the modal, so a text match on it would pass
    // even if the panel never opened.
    await row.locator("td").first().click();
    const keyInfo = page.getByRole("tab", { name: "Overview", exact: true });
    await expect(keyInfo, "key info panel did not open").toBeVisible({
      timeout: 20_000,
    });
    await expect(page.getByRole("tab", { name: "Settings", exact: true })).toBeVisible();
    await expect(page.getByText("Back to Keys", { exact: false })).toBeVisible();
  });
});
