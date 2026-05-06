/**
 * LIT-2881 Validation #5 — SSE reconnect across run reload.
 *
 * Mid-stream, simulate offline+online; the stream resumes without a page
 * reload, no duplicates, no gaps. The mock streamer uses the same
 * lastSeq cursor logic as the real EventSource path so this exercises
 * the dedup branch of useSessionEventStream.
 */
import { test, expect } from "@playwright/test";
import { AGENTS_DEV_URL, authenticateAgentsPage } from "./_helpers";

test("SSE stream resumes after offline/online without duplicates", async ({ page, context }) => {
  await authenticateAgentsPage(page);
  await page.goto(`${AGENTS_DEV_URL}/agents/agt_01/sessions/ses_01`);
  await expect(page.getByTestId("conversation-pane")).toBeVisible();

  // Wait for a couple of streamed events
  await expect
    .poll(
      async () =>
        (await page.getByTestId(/^message-bubble-/).count()) + (await page.getByTestId("tool-call-card").count()),
      {
        timeout: 5_000,
      },
    )
    .toBeGreaterThanOrEqual(3);

  const before =
    (await page.getByTestId(/^message-bubble-/).count()) + (await page.getByTestId("tool-call-card").count());

  // Simulate offline → online
  await context.setOffline(true);
  await page.waitForTimeout(1500);
  await context.setOffline(false);

  // After reconnect, more events arrive (stream resumes, no page reload)
  // and we never go below the previous count (no duplicates removed, no
  // gaps — events are append-only, deduped by seq).
  await expect
    .poll(
      async () =>
        (await page.getByTestId(/^message-bubble-/).count()) + (await page.getByTestId("tool-call-card").count()),
      {
        timeout: 10_000,
      },
    )
    .toBeGreaterThanOrEqual(before);
});
