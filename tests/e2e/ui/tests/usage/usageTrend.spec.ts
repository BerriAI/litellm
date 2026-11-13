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

const trendPanel = (page: PlaywrightPage): Locator => page.getByTestId("usage-trend");

test.describe("Usage trend", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  const keys: string[] = [];

  test.afterEach(async ({ request }) => {
    for (const token of keys.splice(0)) await deleteVirtualKey(request, token);
  });

  test("charts a model's traffic by one shared metric and bucket, and lists it in the contribution table", async ({
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

    await expect(panel.getByRole("group", { name: "Daily tokens by model" })).toBeVisible();
    await expect(panel.getByRole("list", { name: "Models" })).toContainText(CHAT_MODEL_A);
    const table = panel.getByRole("region", { name: "Contribution by model" }).getByRole("table");
    await expect(table.getByRole("row").filter({ hasText: CHAT_MODEL_A })).toHaveCount(1);
    await expect(table.getByRole("columnheader", { name: "Observed $ / 1M tokens" })).toBeVisible();

    const controls = page.getByRole("group", { name: "Chart options" });
    await expect(controls).toContainText("Applies to Usage trend and Top agents");
    await controls.getByRole("radiogroup", { name: "Metric" }).getByRole("radio", { name: "Spend" }).click();
    const bucketSize = controls.getByRole("radiogroup", { name: "Bucket size" });
    await bucketSize.getByRole("radio", { name: "Monthly" }).click();
    await expect(bucketSize.getByRole("radio", { name: "Monthly" })).toBeChecked();
    await expect(panel.getByRole("group", { name: "Monthly spend by model" })).toBeVisible();
    await expect(page.getByTestId("top-agents")).toContainText("Monthly spend by client");

    await controls.getByRole("radio", { name: "Requests" }).click();
    await expect(panel.getByRole("group", { name: "Monthly requests by model" })).toBeVisible();
    await expect(panel.getByRole("group", { name: "Monthly spend by model" })).toHaveCount(0);
  });
});
