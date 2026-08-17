import { expect, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { navigateToPage, dismissFeedbackPopup } from "./navigation";
import { Page } from "../fixtures/pages";

/** Controls for the Test Key / Playground page, shared with the router-fallback specs. */

/**
 * The configuration panel is rendered twice, docked and overlay, with one visible at a time.
 * Every control is narrowed to the visible copy or it trips strict mode against its hidden twin.
 */
export const onlyVisible = (locator: Locator): Locator => locator.filter({ visible: true }).first();

/** The model combobox, addressed by the placeholder its search input shows before selection. */
export const modelSelect = (page: PlaywrightPage): Locator => onlyVisible(page.getByPlaceholder("Select a Model"));

export const sendButton = (page: PlaywrightPage): Locator =>
  onlyVisible(page.getByRole("button", { name: "Send message" }));

/** The Virtual Key Source dropdown, addressed by the accessible name on its trigger. */
export const keySourceSelect = (page: PlaywrightPage): Locator =>
  onlyVisible(page.getByLabel("Virtual Key Source"));

export async function openPlayground(page: PlaywrightPage): Promise<void> {
  await navigateToPage(page, Page.LlmPlayground);
  await dismissFeedbackPopup(page);
  await expect(onlyVisible(page.getByText("Virtual Key Source"))).toBeVisible({
    timeout: 20_000,
  });
}

export async function selectModel(page: PlaywrightPage, model: string): Promise<void> {
  const select = modelSelect(page);
  await select.click();
  // Virtualized: options outside the rendered window are absent from the DOM, so search first.
  await select.fill(model);
  await onlyVisible(page.getByRole("option", { name: model, exact: true })).click({ timeout: 15_000 });
}

export async function sendMessage(page: PlaywrightPage, message: string): Promise<void> {
  const input = onlyVisible(page.getByPlaceholder("Type your message", { exact: false }));
  await expect(input).toBeVisible({ timeout: 15_000 });
  await input.fill(message);
  await sendButton(page).click();
}
