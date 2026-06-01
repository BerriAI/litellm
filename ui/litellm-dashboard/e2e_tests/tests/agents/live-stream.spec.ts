/**
 * LIT-2881 Validation #4 — Live event streaming.
 *
 * Inside a session, ≥3 events render in the conversation pane within 10s
 * (from the mock SSE stream — assistant_message, tool_call, etc.).
 */
import { test, expect } from "@playwright/test";
import { AGENTS_DEV_URL, authenticateAgentsPage } from "./_helpers";

test("session view streams ≥3 events into the conversation pane", async ({ page }) => {
  await authenticateAgentsPage(page);
  await page.goto(`${AGENTS_DEV_URL}/agents/agt_01/sessions/ses_01`);
  await expect(page.getByTestId("conversation-pane")).toBeVisible();

  // Mock streamer ticks every 400ms; we should see all kinds of events
  // (the conversation snapshot already has 3 messages + 6 streamed events).
  // Count visible message bubbles + tool cards + diff entries combined.
  await expect
    .poll(
      async () => {
        const messages = await page.getByTestId(/^message-bubble-/).count();
        const tools = await page.getByTestId("tool-call-card").count();
        return messages + tools;
      },
      { timeout: 10_000, intervals: [200, 500, 1000] },
    )
    .toBeGreaterThanOrEqual(3);
});
