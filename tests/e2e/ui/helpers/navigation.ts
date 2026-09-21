import { Page } from "../fixtures/pages";
import { Page as PlaywrightPage, expect } from "@playwright/test";
import { UI_BASE_URL } from "../constants";

export const sidebarLink = (page: PlaywrightPage, name: string | RegExp) =>
  page.getByRole("complementary").getByRole("link", { name, exact: true });

export async function clickSidebarLink(page: PlaywrightPage, name: string | RegExp, groupName?: string): Promise<void> {
  const link = sidebarLink(page, name);
  if (groupName && !(await link.isVisible())) {
    const group = page.getByRole("complementary").getByRole("button", { name: groupName, exact: true });
    await expect(group).toBeVisible();
    if ((await group.getAttribute("aria-expanded")) === "false") {
      await group.click();
    }
  }
  await link.click();
}

export async function expectUiRoute(page: PlaywrightPage, segment: string): Promise<void> {
  const root = (process.env.SERVER_ROOT_PATH ?? "").replace(/\/+$/, "");
  const expected = new URL(`${root}/ui/${segment}`, UI_BASE_URL);
  await expect(page, `navigate to ${expected.pathname}`).toHaveURL(
    (url) => url.origin === expected.origin && url.pathname.replace(/\/+$/, "") === expected.pathname,
  );
}

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
 * clickable span (Teams Typography.Text).
 */
export async function clickTeamId(page: PlaywrightPage, teamId: string): Promise<void> {
  const cell = page.locator("td").filter({ hasText: teamId }).first();
  await expect(cell).toBeVisible({ timeout: 10_000 });
  await cell.click();
  await expect(page.getByText("Back to Teams")).toBeVisible({ timeout: 10_000 });
}

export async function openKeyDetail(page: PlaywrightPage, alias: string): Promise<void> {
  await page.getByPlaceholder("Search by key alias or ID").fill(alias);
  const row = page.getByRole("row").filter({ hasText: alias });
  await expect(row, `key row "${alias}" never appeared on the Virtual Keys page`).toBeVisible({ timeout: 15_000 });
  await row.getByRole("button", { name: alias }).click();
  await expect(page.getByText("Back to Keys"), `key detail for "${alias}" never opened`).toBeVisible({
    timeout: 15_000,
  });
}
