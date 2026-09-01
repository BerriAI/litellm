import { test, expect, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { Page } from "../../fixtures/pages";
import {
  CHAT_MODEL_A,
  CHAT_MODEL_B,
  DEPLOYMENT_MODEL_A,
  DEPLOYMENT_MODEL_B,
  createVirtualKey,
  masterKey,
  sendChatCompletion,
  waitForKeyInDailyActivity,
  waitForSpendLog,
} from "../../helpers/traffic";

/**
 * Covers the per-entity breakdowns on /ui/usage. The page-level totals move with every other spec's
 * traffic, so each assertion is scoped to a key this test minted and to the requests it sent.
 */

/** Each breakdown renders one expandable card per entity, named "<entity> $x.xx N requests". */
const entityCard = (page: PlaywrightPage, tab: string, name: string): Locator =>
  page.getByRole("tabpanel", { name: tab }).getByRole("button", { name: new RegExp(`^${name}\\s`) });

async function openUsageTab(page: PlaywrightPage, tab: string): Promise<Locator> {
  await navigateToPage(page, Page.NewUsage);
  await dismissFeedbackPopup(page);
  await page.getByRole("tab", { name: tab }).click();
  const panel = page.getByRole("tabpanel", { name: tab });
  await expect(panel).toBeVisible({ timeout: 30_000 });
  return panel;
}

/** Sends `count` completions on one model and waits for each to reach the spend log. */
async function sendTraffic(
  request: Parameters<typeof sendChatCompletion>[0],
  apiKey: string,
  model: string,
  count: number,
  label: string,
): Promise<void> {
  for (let i = 0; i < count; i++) {
    const requestId = await sendChatCompletion(request, { model, prompt: `${label} ${i}`, apiKey });
    await waitForSpendLog(request, requestId);
  }
}

test.describe("Usage page activity tabs", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Key Activity breaks a key's traffic down by model", async ({ page, request }) => {
    const alias = `e2e-usage-keyact-${Date.now()}`;
    const { key, token } = await createVirtualKey(request, { key_alias: alias });

    // An uneven split, so a breakdown that lumps everything into one row or attributes to the
    // wrong model cannot land on these numbers by accident.
    await sendTraffic(request, key, CHAT_MODEL_A, 2, alias);
    await sendTraffic(request, key, CHAT_MODEL_B, 1, alias);
    await waitForKeyInDailyActivity(request, token, 3);

    await openUsageTab(page, "Key Activity");

    const card = entityCard(page, "Key Activity", alias);
    await expect(card, `${alias} missing from Key Activity`).toBeVisible({ timeout: 30_000 });
    await expect(card).toContainText("3 requests");

    // Every key gets a card, and the page opens the first one. Scope to this key's own section,
    // which the collapsible renders as the trigger's next sibling.
    await card.click();
    const details = card.locator("xpath=following-sibling::*[1]");
    const successfulFor = (model: string) =>
      details.getByRole("row").filter({ hasText: model }).getByRole("cell").nth(2); // Model | Spend | Successful | Failed | Tokens

    await expect(successfulFor(DEPLOYMENT_MODEL_A)).toHaveText("2", { timeout: 20_000 });
    await expect(successfulFor(DEPLOYMENT_MODEL_B)).toHaveText("1");
  });

  test("Model Activity can name its models by deployment instead of by public name", async ({ page, request }) => {
    const alias = `e2e-usage-modelact-${Date.now()}`;
    const { key, token } = await createVirtualKey(request, { key_alias: alias });
    await sendTraffic(request, key, CHAT_MODEL_A, 1, alias);
    await waitForKeyInDailyActivity(request, token);

    const panel = await openUsageTab(page, "Model Activity");

    await expect(entityCard(page, "Model Activity", CHAT_MODEL_A), `${CHAT_MODEL_A} missing`).toBeVisible({
      timeout: 30_000,
    });
    // Nothing is published under the deployment's name, so its absence here is what makes the
    // toggle below a real change of key rather than a relabelled button.
    await expect(entityCard(page, "Model Activity", DEPLOYMENT_MODEL_A)).toHaveCount(0);

    // Admins reconcile provider bills against the deployment, not the name their users call.
    await panel.getByRole("button", { name: "Litellm Model Name" }).click();
    await expect(entityCard(page, "Model Activity", DEPLOYMENT_MODEL_A)).toBeVisible({ timeout: 20_000 });
  });

  test("Filter by user narrows Key Activity to that user's keys", async ({ page, request }) => {
    const stamp = Date.now();
    const email = `e2e-usage-owner-${stamp}@test.local`;
    const ownedAlias = `e2e-usage-owned-${stamp}`;
    const otherAlias = `e2e-usage-other-${stamp}`;

    const userRes = await request.post("/user/new", {
      headers: { Authorization: `Bearer ${masterKey()}`, "Content-Type": "application/json" },
      data: { user_email: email, user_role: "internal_user", auto_create_key: false },
    });
    expect(userRes.ok(), `POST /user/new failed (${userRes.status()})`).toBe(true);
    const userId = (await userRes.json()).user_id as string;

    const owned = await createVirtualKey(request, { key_alias: ownedAlias, user_id: userId });
    const other = await createVirtualKey(request, { key_alias: otherAlias });
    await sendTraffic(request, owned.key, CHAT_MODEL_A, 1, ownedAlias);
    await sendTraffic(request, other.key, CHAT_MODEL_A, 1, otherAlias);
    await waitForKeyInDailyActivity(request, owned.token);
    await waitForKeyInDailyActivity(request, other.token);

    await openUsageTab(page, "Key Activity");
    await expect(entityCard(page, "Key Activity", otherAlias)).toBeVisible({ timeout: 30_000 });

    await page.getByRole("combobox", { name: "Search users by email" }).click();
    await page.keyboard.type(email);
    await page
      .getByRole("option", { name: new RegExp(email) })
      .first()
      .click();

    // The filter earns its place only by dropping the other key; the owned key showing up
    // proves nothing on a page that already listed every key.
    await expect(entityCard(page, "Key Activity", otherAlias)).toHaveCount(0, { timeout: 30_000 });
    await expect(entityCard(page, "Key Activity", ownedAlias)).toBeVisible({ timeout: 20_000 });
  });
});
