import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";
import { masterKey } from "../../helpers/traffic";

interface StoredBudget {
  budget_id: string;
  max_budget: number | null;
  tpm_limit: number | null;
  rpm_limit: number | null;
  budget_duration: string | null;
}

/** A different route from the one the table renders from, so a row that only lives in its cache fails here. */
async function findBudget(page: PlaywrightPage, budgetId: string): Promise<StoredBudget | undefined> {
  const res = await page.request.get("/budget/list", {
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  expect(res.ok(), `GET /budget/list (${res.status()})`).toBe(true);
  return ((await res.json()) as StoredBudget[]).find((row) => row.budget_id === budgetId);
}

async function createBudgetViaApi(page: PlaywrightPage, budget: Partial<StoredBudget>): Promise<void> {
  const res = await page.request.post("/budget/new", {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: budget,
  });
  expect(res.ok(), `POST /budget/new failed (${res.status()}): ${await res.text()}`).toBe(true);
}

async function searchForBudget(page: PlaywrightPage, budgetId: string): Promise<void> {
  await page.getByPlaceholder("Search by budget ID").fill(budgetId);
}

test.describe("Budgets", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Create a budget with rate limits and a spend cap", async ({ page }) => {
    const budgetId = `e2e-budget-create-${Date.now()}`;

    await navigateToPage(page, Page.Budgets);
    await dismissFeedbackPopup(page);

    await page.getByRole("button", { name: "Create Budget" }).click();

    const modal = page.getByRole("dialog", { name: "Create Budget" });
    await expect(modal).toBeVisible({ timeout: 10_000 });

    await modal.getByRole("textbox", { name: "Budget ID" }).fill(budgetId);
    await modal.getByRole("spinbutton", { name: "Max Tokens per minute" }).fill("5000");
    await modal.getByRole("spinbutton", { name: "Max Requests per minute" }).fill("60");

    await modal.getByRole("button", { name: "Optional Settings" }).click();
    await modal.getByRole("spinbutton", { name: "Max Budget (USD)" }).fill("25.5");
    await modal.getByRole("combobox", { name: "Reset Budget" }).click();
    await page.getByRole("option", { name: "weekly" }).click();

    await modal.getByRole("button", { name: "Create Budget" }).click();
    await expect(modal).not.toBeVisible({ timeout: 10_000 });

    await searchForBudget(page, budgetId);
    const row = page.getByRole("row").filter({ hasText: budgetId });
    await expect(row).toBeVisible({ timeout: 10_000 });
    await expect(row).toContainText("$25.50");

    const stored = await findBudget(page, budgetId);
    expect(stored, `budget ${budgetId} readable from /budget/list`).toBeTruthy();
    expect(stored?.max_budget, "spend cap persisted").toBe(25.5);
    expect(stored?.tpm_limit, "TPM limit persisted").toBe(5000);
    expect(stored?.rpm_limit, "RPM limit persisted").toBe(60);
    expect(stored?.budget_duration, "reset window persisted").toBe("7d");
  });

  test("Raising a budget's spend cap leaves its rate limits alone", async ({ page }) => {
    const budgetId = `e2e-budget-edit-${Date.now()}`;
    await createBudgetViaApi(page, { budget_id: budgetId, max_budget: 10, tpm_limit: 1000, rpm_limit: 20 });

    await navigateToPage(page, Page.Budgets);
    await dismissFeedbackPopup(page);

    await searchForBudget(page, budgetId);
    await expect(page.getByRole("row").filter({ hasText: budgetId })).toBeVisible({ timeout: 10_000 });

    await page.getByTestId(`budget-actions-${budgetId}`).click();
    await page.getByTestId("budget-action-edit").click();

    const modal = page.getByRole("dialog", { name: "Edit Budget" });
    await expect(modal).toBeVisible({ timeout: 10_000 });

    await modal.getByRole("button", { name: "Optional Settings" }).click();
    await modal.getByRole("spinbutton", { name: "Max Budget (USD)" }).fill("99");
    await modal.getByRole("button", { name: "Save", exact: true }).click();
    await expect(modal).not.toBeVisible({ timeout: 10_000 });

    await expect(page.getByRole("row").filter({ hasText: budgetId })).toContainText("$99.00", { timeout: 10_000 });

    // Not hypothetical: the edit form posts the whole budget, so a field it fails to
    // seed from the existing row goes to the server as null and silently clears.
    const stored = await findBudget(page, budgetId);
    expect(stored?.max_budget, "spend cap raised").toBe(99);
    expect(stored?.tpm_limit, "TPM limit untouched by a spend-cap edit").toBe(1000);
    expect(stored?.rpm_limit, "RPM limit untouched by a spend-cap edit").toBe(20);
  });

  test("Delete a budget", async ({ page }) => {
    const budgetId = `e2e-budget-delete-${Date.now()}`;
    await createBudgetViaApi(page, { budget_id: budgetId, max_budget: 5 });

    await navigateToPage(page, Page.Budgets);
    await dismissFeedbackPopup(page);

    await searchForBudget(page, budgetId);
    await expect(page.getByRole("row").filter({ hasText: budgetId })).toBeVisible({ timeout: 10_000 });

    await page.getByTestId(`budget-actions-${budgetId}`).click();
    await page.getByTestId("budget-action-delete").click();

    const modal = page.getByRole("dialog", { name: "Delete Budget?" });
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.getByRole("button", { name: "Delete", exact: true }).click();

    await expect(page.getByRole("row").filter({ hasText: budgetId })).toHaveCount(0, { timeout: 10_000 });

    // The row disappearing is a cache invalidation; the budget is gone when the route stops serving it.
    await expect
      .poll(async () => await findBudget(page, budgetId), {
        message: `budget ${budgetId} still readable from /budget/list after delete`,
        timeout: 15_000,
      })
      .toBeUndefined();
  });
});
