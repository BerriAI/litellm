/**
 * LIT-2881 Validation #6 — Followup composer.
 *
 * Type message + send. user_message bubble appears (mock provider).
 * (assistant_message follows in real proxy mode; the mock client only
 * acks the user_message synchronously, which is enough to verify the
 * composer plumbing.)
 */
import { test, expect } from "@playwright/test";
import { AGENTS_DEV_URL, authenticateAgentsPage } from "./_helpers";

test("composer send surfaces a user_message in the conversation pane", async ({ page }) => {
  await authenticateAgentsPage(page);
  await page.goto(`${AGENTS_DEV_URL}/agents/agt_01/sessions/ses_01`);
  await expect(page.getByTestId("composer")).toBeVisible();

  const userBubblesBefore = await page.getByTestId("message-bubble-user").count();

  await page.getByTestId("composer-input").fill("ping from e2e");
  await page.getByTestId("composer-send").click();

  // user_message bubble count strictly grows
  await expect
    .poll(async () => page.getByTestId("message-bubble-user").count(), { timeout: 5_000 })
    .toBeGreaterThan(userBubblesBefore);
});
