/**
 * LIT-2881 Validation #7 — Terminal tab renders ANSI.
 *
 * Mock proxy emits a `terminal_chunk` with ANSI red text (\x1b[31m).
 * The TerminalTab parses SGR sequences and emits a span with
 * data-testid="ansi-#ff0000" and computed color rgb(255, 0, 0).
 */
import { test, expect } from "@playwright/test";
import { AGENTS_DEV_URL, authenticateAgentsPage } from "./_helpers";

test("terminal tab renders ANSI red as computed color rgb(255, 0, 0)", async ({ page }) => {
  await authenticateAgentsPage(page);
  await page.goto(`${AGENTS_DEV_URL}/agents/agt_01/sessions/ses_01`);

  // Switch to the Terminal tab
  await page.getByRole("tab", { name: "Terminal" }).click();
  await expect(page.getByTestId("terminal-tab")).toBeVisible();

  // Wait for the streamed terminal_chunk to arrive (mock ticks every 400ms)
  const redSpan = page.getByTestId("ansi-#ff0000").first();
  await expect(redSpan).toBeVisible({ timeout: 10_000 });

  const computed = await redSpan.evaluate((el) => window.getComputedStyle(el).color);
  expect(computed).toBe("rgb(255, 0, 0)");
});
