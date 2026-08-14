import { expect, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { navigateToPage, dismissFeedbackPopup } from "./navigation";
import { Page } from "../fixtures/pages";

/** Controls for the Test Key / Playground page, shared with the router-fallback specs. */

/**
 * The configuration panel is rendered twice, docked and overlay, with one visible at a time.
 * Every control is narrowed to the visible copy or it trips strict mode against its hidden twin.
 */
export const onlyVisible = (locator: Locator): Locator => locator.filter({ visible: true }).first();

/** The model dropdown, addressed by the placeholder it shows before selection. */
export const modelSelect = (page: PlaywrightPage): Locator =>
  onlyVisible(page.locator('.ant-select:has(.ant-select-selection-placeholder:text-is("Select a Model"))'));

/** Send button is icon-only (an up-arrow), so there is no accessible name. */
export const sendButton = (page: PlaywrightPage): Locator => onlyVisible(page.locator("button:has(.anticon-arrow-up)"));

/** The Virtual Key Source dropdown, addressed by its currently selected label. */
export const keySourceSelect = (page: PlaywrightPage, current: string): Locator =>
  onlyVisible(page.locator(`.ant-select:has(.ant-select-selection-item[title="${current}"])`));

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
  await select.locator("input.ant-select-selection-search-input").fill(model);
  // antd portals its dropdown to the body; options carry the value as `title`.
  await onlyVisible(page.locator(`.ant-select-item-option[title="${model}"]`)).click({ timeout: 15_000 });
}

export async function sendMessage(page: PlaywrightPage, message: string): Promise<void> {
  const input = onlyVisible(page.getByPlaceholder("Type your message", { exact: false }));
  await expect(input).toBeVisible({ timeout: 15_000 });
  await input.fill(message);
  await sendButton(page).click();
}
