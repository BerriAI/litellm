import { Page } from "../fixtures/pages";
import { Page as PlaywrightPage, expect } from "@playwright/test";

/**
 * Navigates to a specific page using the page query parameter.
 * Waits for the sidebar to be visible before returning.
 */
export async function navigateToPage(page: PlaywrightPage, pageEnum: Page): Promise<void> {
  await page.goto(`/ui?page=${pageEnum}`);
  await page.waitForLoadState("networkidle");
  // Dismiss the "Quick feedback" popup if it appears
  await dismissFeedbackPopup(page);
}

/**
 * Dismiss the "Quick feedback" popup that may appear on any page.
 */
export async function dismissFeedbackPopup(page: PlaywrightPage): Promise<void> {
  const dismissButton = page.getByText("Don't ask me again");
  if (await dismissButton.isVisible({ timeout: 1_500 }).catch(() => false)) {
    await dismissButton.click();
    // Wait for the popup to disappear
    await expect(dismissButton).not.toBeVisible({ timeout: 2_000 }).catch(() => {});
  }
}
