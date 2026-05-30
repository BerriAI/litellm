import { test, expect } from "@playwright/test";
import {
  E2E_TEAM_CRUD_ID,
  E2E_VIEWER_KEY_ALIAS,
  INTERNAL_VIEWER_STORAGE_PATH,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";

async function clickTeamId(page: import("@playwright/test").Page, teamId: string) {
  const cell = page.locator("td").filter({ hasText: teamId }).first();
  await expect(cell).toBeVisible({ timeout: 10_000 });
  await cell.click();
  await expect(page.getByText("Back to Teams")).toBeVisible({ timeout: 10_000 });
}

test.describe("Internal Viewer", () => {
  test.use({ storageState: INTERNAL_VIEWER_STORAGE_PATH });

  test("Nav shows only the allowed options for the Internal Viewer role", async ({ page }) => {
    // Use navigateToPage so the networkidle wait lets the async role-gated nav
    // settle before we assert — a bare page.goto races the permission fetch.
    await navigateToPage(page, Page.ApiKeys);

    // Scope to the sidebar and match items by their link role + accessible
    // name. The sidebar is a `complementary` landmark (the `navigation` role
    // is the top bar), and each item renders as a link inside it — far tighter
    // than a CSS `nav, aside` selector or a getByText on stray text nodes.
    const nav = page.getByRole("complementary");

    // Items that must be visible per the manual-QA checklist
    const expectedVisible = [
      "Virtual Keys",
      "MCP Servers",
      "Guardrails",
      "Usage",
      "Logs",
      "Teams",
      "API Reference",
      "AI Hub",
    ];
    for (const label of expectedVisible) {
      await expect(
        nav.getByRole("link", { name: label, exact: true }).first(),
        `expected nav item "${label}" to render for Internal Viewer`,
      ).toBeVisible({ timeout: 5_000 });
    }

    // Items that must NOT be visible (admin-only surface)
    const expectedHidden = ["Internal Users", "Organizations", "Models + Endpoints"];
    for (const label of expectedHidden) {
      await expect(
        nav.getByRole("link", { name: label, exact: true }),
        `nav item "${label}" must not render for Internal Viewer`,
      ).toHaveCount(0);
    }
  });

  test("Virtual Keys page hides Create / Regenerate / Reset / Delete controls", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    // Create button is gated on rolesWithWriteAccess (Internal Viewer is not in it)
    await expect(page.getByRole("button", { name: /Create New Key/i })).toHaveCount(0);

    // Open the viewer's own key info page
    const keyRow = page.locator("tr", { hasText: E2E_VIEWER_KEY_ALIAS });
    await expect(keyRow).toBeVisible({ timeout: 10_000 });
    await keyRow.locator("button").first().click();
    await expect(page.getByText("Back to Keys")).toBeVisible({ timeout: 10_000 });

    // None of the destructive / mutating actions should render
    await expect(page.getByRole("button", { name: "Regenerate Key" })).toHaveCount(0);
    await expect(page.getByRole("button", { name: /Reset Spend/i })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Delete Key" })).toHaveCount(0);
  });

  test("Team info page omits Members and Settings tabs for an Internal Viewer", async ({ page }) => {
    await navigateToPage(page, Page.Teams);

    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    // Overview / Virtual Keys are always visible; Settings + Members are not.
    // Tabs are conditionally rendered (getTeamInfoVisibleTabs filters the list),
    // so assert absence from the DOM with toHaveCount(0) to match the nav block.
    await expect(page.getByRole("tab", { name: "Overview" })).toBeVisible({ timeout: 5_000 });
    await expect(page.getByRole("tab", { name: "Virtual Keys" })).toBeVisible({ timeout: 5_000 });
    await expect(page.getByRole("tab", { name: "Settings" })).toHaveCount(0);
    await expect(page.getByRole("tab", { name: "Members" })).toHaveCount(0);
  });
});
