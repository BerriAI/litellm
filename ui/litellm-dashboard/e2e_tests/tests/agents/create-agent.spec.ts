/**
 * LIT-2881 Validation #2 — Create an Agent definition flow.
 *
 * Click + New Agent, fill name/model/system_prompt, submit; the new row
 * appears in the /agents list within 5s. Definition only — no VM.
 */
import { test, expect } from "@playwright/test";
import { gotoAgentsList } from "./_helpers";

test("create new agent definition appears in list", async ({ page }) => {
  await gotoAgentsList(page);
  await page.getByTestId("new-agent-btn").click();
  const dialog = page.getByTestId("new-agent-dialog");
  await expect(dialog).toBeVisible();

  const nameInput = page.getByTestId("new-agent-name").locator("input");
  const modelInput = page.getByTestId("new-agent-model").locator("input");
  await nameInput.fill("e2e-test-agent");
  await modelInput.fill("claude-3-5-sonnet-20241022");

  await page.getByRole("button", { name: "Create" }).click();
  await expect(dialog).not.toBeVisible({ timeout: 5_000 });

  // appears in the list (within 5s)
  await expect(page.getByText("e2e-test-agent")).toBeVisible({ timeout: 5_000 });
});
