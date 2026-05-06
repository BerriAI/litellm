/**
 * LIT-2881 Validation #3 — Create a Session under an Agent.
 *
 * From /agents/{aid}, click + New Session, fill repo URL; submit redirects
 * to /agents/{aid}/sessions/{new-sid}; status pill = provisioning.
 */
import { test, expect } from "@playwright/test";
import { AGENTS_DEV_URL, authenticateAgentsPage } from "./_helpers";

test("create new session redirects to three-pane with provisioning pill", async ({ page }) => {
  await authenticateAgentsPage(page);
  await page.goto(`${AGENTS_DEV_URL}/agents/agt_01`);
  await page.getByTestId("agent-detail-new-session-btn").click();

  const dialog = page.getByTestId("new-session-dialog");
  await expect(dialog).toBeVisible();
  await page.getByTestId("new-session-repo-url").locator("input").fill("https://github.com/example/new-repo");
  await page.getByRole("button", { name: "Create session" }).click();

  // Redirects into the new three-pane URL
  await page.waitForURL(/\/agents\/agt_01\/sessions\/ses_/, { timeout: 5_000 });
  await expect(page.getByTestId("three-pane")).toBeVisible();

  // Status pill = provisioning
  const sessionId = new URL(page.url()).pathname.split("/").pop()!;
  const row = page.getByTestId(`session-row-${sessionId}`);
  await expect(row).toBeVisible();
  await expect(row.getByText("provisioning")).toBeVisible();
});
