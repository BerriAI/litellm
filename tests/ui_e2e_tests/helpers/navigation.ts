import { Page as PlaywrightPage } from "@playwright/test";
import { Page } from "../constants";

export async function navigateToPage(page: PlaywrightPage, targetPage: Page) {
  await page.goto(`/ui?page=${targetPage}`);
}
