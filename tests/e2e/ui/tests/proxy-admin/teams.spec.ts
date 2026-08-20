import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import {
  ADMIN_STORAGE_PATH,
  E2E_TEAM_CRUD_ID,
  E2E_TEAM_DELETE_ALIAS,
  E2E_TEAM_NO_ADMIN_ID,
  E2E_TEAM_ORG_ID,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup, clickTeamId } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";

/** GET /team/list returns a bare array of teams, each carrying team_alias/team_id. */
async function findTeamByAlias(page: PlaywrightPage, alias: string): Promise<Record<string, any> | undefined> {
  const teams = await readBack<Record<string, any>[]>(page, "/team/list");
  return teams.find((team) => team.team_alias === alias);
}

/** GET /team/info nests the record under `team_info`; membership lives in members_with_roles. */
async function teamMemberEmails(page: PlaywrightPage, teamId: string): Promise<string[]> {
  const info = await readBack<{ team_info: { members_with_roles?: { user_email?: string }[] } }>(
    page,
    `/team/info?team_id=${encodeURIComponent(teamId)}`,
  );
  return (info.team_info.members_with_roles ?? []).map((member) => member.user_email ?? "").filter(Boolean);
}

test.describe("Proxy Admin - Teams", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Create a team", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    const uniqueAlias = `e2e-created-team-${Date.now()}`;

    // Click the Create Team button — accessible name includes "Create Team"
    await page
      .getByRole("button", { name: /Create Team/i })
      .first()
      .click();

    // Wait for the Create Team modal
    const dialog = page.getByRole("dialog", { name: "Create Team" });
    await expect(dialog).toBeVisible({ timeout: 5_000 });

    // Fill Team Name — FormField derives the control id from React.useId(), so
    // the input is only addressable by its label or its test id.
    await dialog.getByTestId("team-name-input").fill(uniqueAlias);

    // Select models — the models multi-select is inside the modal. Its popup is
    // portaled to the body, so scope the option lookup to the page, not the dialog.
    await dialog.getByTestId("create-team-models-select").getByRole("combobox").click();
    await page.getByRole("option", { name: "All Proxy Models", exact: true }).click();
    await page.keyboard.press("Escape");

    // Submit — click the submit button inside the dialog (not the header button)
    await dialog.locator("button[type='submit']").click();

    // Verify success notification
    await expect(page.getByText("Team created").first()).toBeVisible({ timeout: 10_000 });

    // A create that drops its model selection still toasts success.
    const created = await findTeamByAlias(page, uniqueAlias);
    expect(created, `team ${uniqueAlias} readable from /team/list`).toBeTruthy();
    expect(created?.models, "created team kept its model selection").toBeTruthy();
  });

  test("Invite a user to a team", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    await page.getByRole("tab", { name: "Members" }).click();
    await page.getByRole("button", { name: /Add Member/i }).click();

    // Wait for Add Team Member modal
    const modal = page.getByRole("dialog", { name: "Add Team Member" });
    await expect(modal).toBeVisible({ timeout: 5_000 });

    // The email field is a Select — type to search, then select from dropdown
    await modal.getByRole("combobox").first().click();
    await page.keyboard.type("invitable@test.local");

    // Wait for the option to appear, then select via keyboard (avoids viewport issues)
    const emailOption = page.getByRole("option", { name: "invitable@test.local" }).first();
    await expect(emailOption).toBeAttached({ timeout: 10_000 });
    // Use keyboard to select the highlighted option
    await page.keyboard.press("Enter");

    // Submit
    await modal.getByRole("button", { name: /Add Member/i }).click();

    await expect(page.getByText(/member.*added|success/i).first()).toBeVisible({ timeout: 10_000 });

    // The toast is matched loosely enough (/success/i) that almost any notification satisfies it.
    await expect
      .poll(async () => await teamMemberEmails(page, E2E_TEAM_CRUD_ID), {
        message: "invited user never appeared in the team's members",
        timeout: 15_000,
      })
      .toContain("invitable@test.local");
  });

  test("Edit team member for team proxy admin does not belong to", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_NO_ADMIN_ID);

    await page.getByRole("tab", { name: "Members" }).click();

    await page.getByTestId("edit-member").first().click();

    const modal = page.getByRole("dialog", { name: "Edit Member" });
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.getByRole("button", { name: /Save Changes/i }).click();

    await expect(page.getByText(/updated|success/i).first()).toBeVisible({ timeout: 10_000 });
  });

  test("Delete a team", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    const teamRow = page.locator("tr", { hasText: E2E_TEAM_DELETE_ALIAS }).first();
    await expect(teamRow).toBeVisible({ timeout: 10_000 });
    // Actions live in a kebab menu: open it, then click "Delete team".
    await teamRow.locator('[data-testid^="team-actions-"]').click();
    await page.getByTestId("team-action-delete").click();

    const modal = page.getByRole("dialog", { name: "Delete Team?" });
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.locator("input").fill(E2E_TEAM_DELETE_ALIAS);
    await modal.getByRole("button", { name: /Force Delete|Delete/i }).click();

    await expect(teamRow).not.toBeVisible({ timeout: 10_000 });

    // A row vanishing is local state, which happens whether or not the delete landed.
    await expect
      .poll(async () => await findTeamByAlias(page, E2E_TEAM_DELETE_ALIAS), {
        message: `team ${E2E_TEAM_DELETE_ALIAS} still readable from /team/list after delete`,
        timeout: 15_000,
      })
      .toBeUndefined();
  });

  test("Team in org - edit team member", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_ORG_ID);

    await page.getByRole("tab", { name: "Members" }).click();

    await page.getByTestId("edit-member").first().click();

    const modal = page.getByRole("dialog", { name: "Edit Member" });
    await expect(modal).toBeVisible({ timeout: 5_000 });
    await modal.getByRole("button", { name: /Save Changes/i }).click();

    await expect(page.getByText(/updated|success/i).first()).toBeVisible({ timeout: 10_000 });
  });

  test("Edit team model selection", async ({ page, request }) => {
    // Restore the seeded models via API in case a prior run (or a CI retry)
    // left this team mutated — the assertion below requires fake-anthropic-claude
    // to be present.
    const masterKey = process.env.LITELLM_MASTER_KEY || "sk-1234";
    const seededModels = ["fake-openai-gpt-4", "fake-anthropic-claude"];
    const restore = async () => {
      const res = await request.post("/team/update", {
        headers: { Authorization: `Bearer ${masterKey}` },
        data: { team_id: E2E_TEAM_CRUD_ID, models: seededModels },
      });
      expect(res.ok(), `restore failed: ${res.status()} ${await res.text()}`).toBeTruthy();
    };
    await restore();

    try {
      await navigateToPage(page, Page.Teams);
      await dismissFeedbackPopup(page);

      await clickTeamId(page, E2E_TEAM_CRUD_ID);

      await page.getByRole("tab", { name: "Settings" }).click();
      await page.getByRole("button", { name: "Edit Settings" }).click();

      // Remove the anthropic tag — other tests against this team use "All Team
      // Models" so they pick up whatever remains.
      const modelsSelect = page.locator("[data-testid='models-select']");
      await expect(modelsSelect).toBeVisible({ timeout: 10_000 });

      const anthropicChip = modelsSelect
        .locator('[data-slot="combobox-chip"]')
        .filter({ hasText: "fake-anthropic-claude" });
      await expect(anthropicChip).toBeVisible({ timeout: 5_000 });
      await anthropicChip.locator('[data-slot="combobox-chip-remove"]').click();

      await page.getByRole("button", { name: "Save Changes" }).click();

      await expect(page.getByText(/Team settings updated|updated successfully/i).first()).toBeVisible({
        timeout: 10_000,
      });
    } finally {
      // Leave the team in its seeded state for any subsequent test or rerun.
      await restore();
    }
  });
});
