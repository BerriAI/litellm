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

/** Covers /ui/usage. The legacy /ui/old-usage view is deprecated and deliberately not covered. */

/** Stepping up from the title is exact; the page renders several other tables. */
const topKeysCard = (page: PlaywrightPage): Locator =>
  page.getByText("Top Virtual Keys", { exact: true }).locator("xpath=..");

async function openUsage(page: PlaywrightPage): Promise<Locator> {
  await navigateToPage(page, Page.NewUsage);
  await dismissFeedbackPopup(page);
  const card = topKeysCard(page);
  await expect(card).toBeVisible({ timeout: 30_000 });
  // Widen past the default top-5 so other keys in the database cannot crowd this one out.
  // The radio itself is sr-only and its label covers it, so click the label.
  await card.getByRole("radiogroup", { name: "Number of top keys to show" }).getByText("50", { exact: true }).click();
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
    // The alias is already in the row behind the modal, so match the panel's own controls.
    await row.locator("td").first().click();
    const keyInfo = page.getByRole("tab", { name: "Overview", exact: true });
    await expect(keyInfo, "key info panel did not open").toBeVisible({
      timeout: 20_000,
    });
    await expect(page.getByRole("tab", { name: "Settings", exact: true })).toBeVisible();
    await expect(page.getByText("Back to Keys", { exact: false })).toBeVisible();
  });
});
