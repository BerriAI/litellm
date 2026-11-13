import { test, expect, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import {
  CHAT_MODEL_A,
  createVirtualKey,
  deleteVirtualKey,
  sendChatCompletion,
  uniqueSuffix,
  waitForKeyInDailyActivity,
  waitForSpendLog,
} from "../../helpers/traffic";

/** Covers the Usage trend panel on /ui/usage: paired token and spend charts and the contribution table. */

const trendPanel = (page: PlaywrightPage): Locator => page.getByTestId("usage-trend");

test.describe("Usage trend", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  const keys: string[] = [];

  test.afterEach(async ({ request }) => {
    for (const token of keys.splice(0)) await deleteVirtualKey(request, token);
  });

  test("charts a model's traffic as tokens over spend and lists it in the contribution table", async ({
    page,
    request,
  }) => {
    const alias = `e2e-usage-trend-${uniqueSuffix()}`;
    const { key, token } = await createVirtualKey(request, { key_alias: alias });
    keys.push(token);
    const requestId = await sendChatCompletion(request, {
      model: CHAT_MODEL_A,
      prompt: `trend ${alias}`,
      apiKey: key,
    });
    await waitForSpendLog(request, requestId);
    await waitForKeyInDailyActivity(request, token);

    await navigateToPage(page, Page.NewUsage);
    await dismissFeedbackPopup(page);
    const panel = trendPanel(page);
    await expect(panel.getByRole("heading", { name: "Usage trend" })).toBeVisible({ timeout: 30_000 });

    await expect(panel.getByRole("group", { name: "Tokens stacked by model" })).toBeVisible();
    await expect(panel.getByRole("group", { name: "Spend stacked by model" })).toBeVisible();
    const table = panel.getByRole("region", { name: "Contribution by model" }).getByRole("table");
    await expect(table.getByRole("row").filter({ hasText: CHAT_MODEL_A })).toHaveCount(1);
    await expect(table.getByRole("columnheader", { name: "Observed $ / 1M tokens" })).toBeVisible();

    const bucketSize = panel.getByRole("radiogroup", { name: "Bucket size" });
    await bucketSize.getByRole("radio", { name: "Monthly" }).click();
    await expect(bucketSize.getByRole("radio", { name: "Monthly" })).toBeChecked();
    await expect(panel.getByRole("group", { name: "Spend stacked by model" })).toBeVisible();

    await panel.getByRole("radiogroup", { name: "Chart view" }).getByRole("radio", { name: "Requests" }).click();
    await expect(panel.getByRole("group", { name: "Requests stacked by model" })).toBeVisible();
    await expect(panel.getByRole("group", { name: "Spend stacked by model" })).toHaveCount(0);
  });
});
