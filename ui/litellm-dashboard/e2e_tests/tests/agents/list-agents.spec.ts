/**
 * LIT-2881 Validation #1 — Routes load.
 *
 * Open /agents, /agents/{aid}, /agents/{aid}/sessions/{sid} and verify they
 * each render their primary container without console errors.
 */
import { test, expect } from "@playwright/test";
import { AGENTS_DEV_URL, authenticateAgentsPage } from "./_helpers";

test.describe.configure({ mode: "serial" });

test("/agents lists Agent definitions", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") consoleErrors.push(msg.text());
  });
  await authenticateAgentsPage(page);
  await page.goto(`${AGENTS_DEV_URL}/agents`);
  await expect(page.getByTestId("agents-table")).toBeVisible();
  await expect(page.getByText("shin-cursor-default")).toBeVisible();
  expect(consoleErrors, `console errors: ${consoleErrors.join("\n")}`).toHaveLength(0);
});

test("/agents/{aid} lists sessions under that agent", async ({ page }) => {
  await authenticateAgentsPage(page);
  await page.goto(`${AGENTS_DEV_URL}/agents/agt_01`);
  await expect(page.getByTestId("agent-detail")).toBeVisible();
  await expect(page.getByTestId("agent-detail-sessions")).toBeVisible();
  await expect(page.getByText("Refactor router fallback logic")).toBeVisible();
});

test("/agents/{aid}/sessions/{sid} renders three-pane view", async ({ page }) => {
  await authenticateAgentsPage(page);
  await page.goto(`${AGENTS_DEV_URL}/agents/agt_01/sessions/ses_01`);
  await expect(page.getByTestId("three-pane")).toBeVisible();
  await expect(page.getByTestId("session-list")).toBeVisible();
  await expect(page.getByTestId("conversation-pane")).toBeVisible();
  await expect(page.getByTestId("right-panel")).toBeVisible();
});
