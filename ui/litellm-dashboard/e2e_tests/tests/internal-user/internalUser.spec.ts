import { test, expect } from "@playwright/test";
import {
  E2E_INTERNAL_USER_KEY_ALIAS,
  E2E_TEAM_CRUD_ALIAS,
  E2E_TEAM_CRUD_ID,
  INTERNAL_USER_STORAGE_PATH,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, clickTeamId } from "../../helpers/navigation";

test.describe("Internal User", () => {
  test.use({ storageState: INTERNAL_USER_STORAGE_PATH });

  test("Create Key modal shows the team dropdown populated with the user's teams", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    await page.getByRole("button", { name: /Create New Key/i }).click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    // Open the team dropdown — seeded internal user is a member of
    // e2e-team-crud and e2e-team-org, so we expect at least the CRUD alias.
    const teamSelect = page.locator(".ant-select", { hasText: "Search or select a team" });
    await teamSelect.click();
    await page.keyboard.type(E2E_TEAM_CRUD_ALIAS);
    await expect(
      page.locator(".ant-select-dropdown:visible").getByText(E2E_TEAM_CRUD_ALIAS).first(),
    ).toBeVisible({ timeout: 5_000 });
  });

  test("Team info page omits the Settings tab for non-admin members", async ({ page }) => {
    await navigateToPage(page, Page.Teams);

    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    // Overview / My User / Virtual Keys are always visible; Settings is gated
    // on canEditTeam and must NOT render for a regular team member.
    await expect(page.getByRole("tab", { name: "Overview" })).toBeVisible({ timeout: 5_000 });
    await expect(page.getByRole("tab", { name: "Settings" })).not.toBeVisible();
    await expect(page.getByRole("tab", { name: "Members" })).not.toBeVisible();
  });

  test("Virtual Keys page does not surface litellm-dashboard team keys", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);

    // Anchor on the user's own seeded key so the absence check below cannot
    // pass vacuously against an empty table.
    await expect(
      page.locator("table tbody").getByText(E2E_INTERNAL_USER_KEY_ALIAS).first(),
    ).toBeVisible({ timeout: 10_000 });

    // The litellm-dashboard team is the proxy's internal bookkeeping team —
    // its keys must never leak into an internal user's Virtual Keys table.
    await expect(page.locator("table tbody").getByText("litellm-dashboard")).toHaveCount(0);
  });
});
