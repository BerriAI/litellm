import { Page } from "../fixtures/pages";
import { Page as PlaywrightPage } from "@playwright/test";

/**
 * Navigates to a specific page using the page query parameter.
 * Uses relative path which will be resolved against the baseURL configured in playwright.config.ts
 * @param page - The Playwright page object
 * @param pageEnum - The page enum value to navigate to
 */
export async function navigateToPage(page: PlaywrightPage, pageEnum: Page): Promise<void> {
  await page.goto(`/ui?page=${pageEnum}`);
}
