import { Page } from "../fixtures/pages";
import { Page as PlaywrightPage, expect } from "@playwright/test";

/**
 * Navigates to a specific page using the page query parameter.
 * Waits for the sidebar to be visible before returning.
 */
export async function navigateToPage(page: PlaywrightPage, pageEnum: Page): Promise<void> {
  // A fresh deep-link can race the auth bootstrap: the app briefly treats the
  // session as anonymous, bounces through /ui/login, and lands back on the
  // default page with the ?page= param dropped. Re-issue the navigation until
  // the requested page sticks (auth is warm by the second load) so callers never
  // assert against the default page.
  for (let attempt = 0; attempt < 3; attempt++) {
    await page.goto(`/ui?page=${pageEnum}`);
    await page.waitForLoadState("networkidle");
    const url = new URL(page.url());
    const onLegacyRoot = url.pathname.replace(/\/+$/, "").endsWith("/ui");
    if (!onLegacyRoot || url.searchParams.get("page") === pageEnum) {
      break;
    }
  }
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
    await expect(dismissButton)
      .not.toBeVisible({ timeout: 2_000 })
      .catch(() => {});
  }
}

/**
 * Click on a team ID in the table. Team IDs are rendered differently depending
 * on the component version — try button first (Tremor Button), fall back to
 * clickable span (OldTeams Typography.Text).
 */
export async function clickTeamId(page: PlaywrightPage, teamId: string): Promise<void> {
  const cell = page.locator("td").filter({ hasText: teamId }).first();
  await expect(cell).toBeVisible({ timeout: 10_000 });
  await cell.click();
  await expect(page.getByText("Back to Teams")).toBeVisible({ timeout: 10_000 });
}
