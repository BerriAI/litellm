import { expect, type Locator, type Page as PlaywrightPage } from "@playwright/test";
import { navigateToPage, dismissFeedbackPopup } from "./navigation";
import { Page } from "../fixtures/pages";

export const onlyVisible = (locator: Locator): Locator => locator.filter({ visible: true }).first();

export const composer = (page: PlaywrightPage): Locator =>
  onlyVisible(page.getByRole("group").filter({ has: page.getByRole("textbox", { name: "Message" }) }));

export const modelSelect = (page: PlaywrightPage): Locator =>
  onlyVisible(page.getByRole("combobox", { name: "Model", exact: true }));

export const sendButton = (page: PlaywrightPage): Locator =>
  composer(page).getByRole("button", { name: "Send message" });

export const keySourceSelect = (page: PlaywrightPage): Locator => onlyVisible(page.getByLabel("Send as"));

export async function openPlayground(page: PlaywrightPage): Promise<void> {
  await navigateToPage(page, Page.LlmPlayground);
  await dismissFeedbackPopup(page);
  await expect(modelSelect(page)).toBeVisible({ timeout: 20_000 });
  const searchHint = page.getByRole("button", { name: "Dismiss search hint" });
  if (await searchHint.isVisible()) {
    await searchHint.click();
  }
}

export async function openRequestSettings(page: PlaywrightPage): Promise<void> {
  await composer(page).getByRole("button", { name: "Settings", exact: true }).click();
  await expect(keySourceSelect(page)).toBeVisible({ timeout: 10_000 });
}

export async function enterVirtualKey(page: PlaywrightPage, key: string): Promise<void> {
  await openRequestSettings(page);
  await keySourceSelect(page).click();
  await onlyVisible(page.getByRole("option", { name: "Virtual key", exact: true })).click({ timeout: 15_000 });
  const keyInput = onlyVisible(page.getByLabel("Virtual key", { exact: true }));
  await expect(keyInput).toBeVisible({ timeout: 10_000 });
  await keyInput.fill(key);
  await page.keyboard.press("Escape");
}

export async function searchModels(page: PlaywrightPage, query: string): Promise<void> {
  await modelSelect(page).click();
  await onlyVisible(page.getByPlaceholder("Search models")).fill(query);
}

export async function selectModel(page: PlaywrightPage, model: string): Promise<void> {
  await searchModels(page, model);
  await onlyVisible(page.getByRole("option", { name: model, exact: true })).click({ timeout: 15_000 });
  await expect(modelSelect(page)).toContainText(model);
}

export async function sendMessage(page: PlaywrightPage, message: string): Promise<void> {
  const input = composer(page).getByRole("textbox", { name: "Message" });
  await expect(input).toBeVisible({ timeout: 15_000 });
  await input.fill(message);
  await sendButton(page).click();
}
