import { test, expect } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  E2E_TEAM_CRUD_ID,
  E2E_TEAM_DELETE_ALIAS,
  E2E_TEAM_NO_ADMIN_ID,
  E2E_TEAM_ORG_ID,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";

/**
 * Click on a team ID in the table. Team IDs are rendered differently depending
 * on the component version — try button first (Tremor Button), fall back to
 * clickable span (OldTeams Typography.Text).
 */
async function clickTeamId(page: import("@playwright/test").Page, teamId: string) {
  const cell = page.locator("td").filter({ hasText: teamId }).first();
  await expect(cell).toBeVisible({ timeout: 10_000 });
  await cell.click();
  await expect(page.getByText("Back to Teams")).toBeVisible({ timeout: 10_000 });
}

test.describe("Proxy Admin - Teams", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Create a team", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    const uniqueAlias = `e2e-created-team-${Date.now()}`;

    // Click the Create Team button — accessible name includes "Create Team"
    await page.getByRole("button", { name: /Create Team/i }).first().click();

    // Wait for the Create Team modal
    const dialog = page.locator(".ant-modal:visible");
    await expect(dialog).toBeVisible({ timeout: 5_000 });

    // Fill Team Name — the input has id="team_alias"
    await dialog.locator("#team_alias").fill(uniqueAlias);

    // Select models — the models multi-select is inside the modal
    // Click to open dropdown, select "All Proxy Models"
    await dialog.locator(".ant-select-selection-overflow").first().click();
    await page.locator(".ant-select-dropdown:visible").getByText("All Proxy Models").click();
    await page.keyboard.press("Escape");

    // Submit — click the submit button inside the dialog (not the header button)
    await dialog.locator("button[type='submit']").click();

    // Verify success notification
    await expect(page.getByText("Team created").first()).toBeVisible({ timeout: 10_000 });
  });

  test("Invite a user to a team", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    await page.getByRole("tab", { name: "Members" }).click();
    await page.getByRole("button", { name: /Add Member/i }).click();

    // Wait for Add Team Member modal
    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });

    // The email field is a Select — type to search, then select from dropdown
    await modal.locator(".ant-select").first().click();
    await page.keyboard.type("invitable@test.local");

    // Wait for the option to appear, then select via keyboard (avoids viewport issues)
    const emailOption = page.getByRole("option", { name: "invitable@test.local" }).first();
    await expect(emailOption).toBeAttached({ timeout: 10_000 });
    // Use keyboard to select the highlighted option
    await page.keyboard.press("Enter");

    // Submit
    await modal.getByRole("button", { name: /Add Member/i }).click();

    await expect(page.getByText(/member.*added|success/i).first()).toBeVisible({ timeout: 10_000 });
  });

  test("Edit team member for team proxy admin does not belong to", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_NO_ADMIN_ID);

    await page.getByRole("tab", { name: "Members" }).click();

    await page.getByTestId("edit-member").first().click();

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.getByRole("button", { name: /Save Changes/i }).click();

    await expect(page.getByText(/updated|success/i).first()).toBeVisible({ timeout: 10_000 });
  });

  test("Delete a team", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    const teamRow = page.locator("tr", { hasText: E2E_TEAM_DELETE_ALIAS }).first();
    await expect(teamRow).toBeVisible({ timeout: 10_000 });
    await teamRow.locator("svg, img").last().click();

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.locator("input").fill(E2E_TEAM_DELETE_ALIAS);
    await modal.getByRole("button", { name: /Force Delete|Delete/i }).click();

    await expect(teamRow).not.toBeVisible({ timeout: 10_000 });
  });

  test("Team in org - edit team member", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_ORG_ID);

    await page.getByRole("tab", { name: "Members" }).click();

    await page.getByTestId("edit-member").first().click();

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.getByRole("button", { name: /Save Changes/i }).click();

    await expect(page.getByText(/updated|success/i).first()).toBeVisible({ timeout: 10_000 });
  });
});
