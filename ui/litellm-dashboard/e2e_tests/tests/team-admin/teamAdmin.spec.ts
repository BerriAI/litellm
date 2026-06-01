import { test, expect } from "@playwright/test";
import {
  E2E_INTERNAL_USER_KEY_ALIAS,
  E2E_TEAM_CRUD_ALIAS,
  E2E_TEAM_CRUD_ID,
  TEAM_ADMIN_STORAGE_PATH,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup } from "../../helpers/navigation";

async function clickTeamId(page: import("@playwright/test").Page, teamId: string) {
  const cell = page.locator("td").filter({ hasText: teamId }).first();
  await expect(cell).toBeVisible({ timeout: 10_000 });
  await cell.click();
  await expect(page.getByText("Back to Teams")).toBeVisible({ timeout: 10_000 });
}

test.describe("Team Admin", () => {
  test.use({ storageState: TEAM_ADMIN_STORAGE_PATH });

  test("Team admin can see all team keys including internal user keys", async ({ page }) => {
    // Step from the manual-QA checklist: navigate into the team info page,
    // open the Virtual Keys tab, and confirm a key belonging to another
    // team member (the seeded internal user) is visible.
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    await page.getByRole("tab", { name: "Virtual Keys" }).click();
    await expect(page.getByText(E2E_INTERNAL_USER_KEY_ALIAS).first())
      .toBeVisible({ timeout: 10_000 });

    // And from the global Virtual Keys page, the same key should be visible.
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByText(E2E_INTERNAL_USER_KEY_ALIAS).first())
      .toBeVisible({ timeout: 10_000 });
  });

  test("Team admin can add a member to their team", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    await page.getByRole("tab", { name: "Members" }).click();
    await page.getByRole("button", { name: /Add Member/i }).click();

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });

    // Use a dedicated invitee user so this doesn't race with the proxy-admin
    // "Invite a user" test that adds invitable@test.local to the same team.
    await modal.locator(".ant-select").first().click();
    await page.keyboard.type("invitable-team@test.local");

    const emailOption = page.getByRole("option", { name: "invitable-team@test.local" }).first();
    await expect(emailOption).toBeAttached({ timeout: 10_000 });
    await page.keyboard.press("Enter");

    await modal.getByRole("button", { name: /Add Member/i }).click();

    await expect(page.getByText("Team member added successfully").first())
      .toBeVisible({ timeout: 10_000 });
  });

  test("Team admin can remove a member from their team", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    await page.getByRole("tab", { name: "Members" }).click();

    // Seeded members appear in the roster by user_id (members_with_roles has no
    // email), so match the row on the user_id rather than the email.
    const row = page.locator("tr", { hasText: "e2e-removable-member" }).first();
    await expect(row).toBeVisible({ timeout: 10_000 });
    await row.getByTestId("delete-member").click();

    const modal = page.locator(".ant-modal:visible");
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.getByRole("button", { name: /^Delete$/ }).click();

    await expect(page.getByText("Team member removed successfully").first())
      .toBeVisible({ timeout: 10_000 });
  });

  test("Team admin can create a team key with All Team Models", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    await page.getByRole("button", { name: /Create New Key/i }).click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    const keyName = `e2e-team-admin-key-${Date.now()}`;
    await page.getByTestId("base-input").fill(keyName);

    // Team selector — same locator pattern as the proxy-admin keys test.
    const teamSelect = page.locator(".ant-select", { hasText: "Search or select a team" });
    await teamSelect.click();
    await page.keyboard.type(E2E_TEAM_CRUD_ALIAS);
    await page.locator(".ant-select-dropdown:visible").getByText(E2E_TEAM_CRUD_ALIAS).first().click();

    // Models — pick "All Team Models"
    await page.locator(".ant-select-selection-overflow").click();
    await page.locator(".ant-select-dropdown:visible").getByText("All Team Models").click();
    await page.keyboard.press("Escape");

    await page.getByRole("button", { name: "Create Key", exact: true }).click();

    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });
    await page.keyboard.press("Escape");

    await expect(page.getByText(keyName)).toBeVisible({ timeout: 10_000 });
  });
});
