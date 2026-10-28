import { Page as PlaywrightPage, expect } from "@playwright/test";
import { Page } from "../fixtures/pages";
import { dismissFeedbackPopup, navigateToPage } from "./navigation";

export async function deleteGuardrailFromTable(page: PlaywrightPage, guardrailName: string): Promise<void> {
  await navigateToPage(page, Page.Guardrails);
  await dismissFeedbackPopup(page);
  const row = page.getByRole("row").filter({ hasText: guardrailName });
  await expect(row).toHaveCount(1, { timeout: 15_000 });
  await row.getByRole("button", { name: "Open guardrail actions" }).click();
  await page.getByRole("menuitem", { name: "Delete" }).click();

  const deleteModal = page.getByRole("dialog", { name: "Delete Guardrail" });
  await expect(deleteModal).toBeVisible({ timeout: 5_000 });
  await deleteModal.getByPlaceholder(guardrailName, { exact: true }).fill(guardrailName);
  await deleteModal.getByRole("button", { name: "Delete", exact: true }).click();

  await expect(page.getByText(`Guardrail "${guardrailName}" deleted successfully`)).toBeVisible({ timeout: 10_000 });
  await expect(row).toHaveCount(0, { timeout: 15_000 });
}
