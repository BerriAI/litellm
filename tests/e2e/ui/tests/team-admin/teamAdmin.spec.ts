import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import {
  E2E_INTERNAL_USER_KEY_ALIAS,
  E2E_TEAM_CRUD_ALIAS,
  E2E_TEAM_CRUD_ID,
  TEAM_ADMIN_STORAGE_PATH,
} from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup, clickTeamId } from "../../helpers/navigation";
import { captureRequestBody, readBack } from "../../helpers/roundTrip";

/**
 * Every identifier a roster is addressable by. Which of user_id / user_email is populated depends on
 * how the member got there, so flatten both and let assertions name whichever the test typed.
 */
async function teamMemberIdentities(page: PlaywrightPage, teamId: string): Promise<string[]> {
  const info = await readBack<{ team_info: { members_with_roles?: { user_id?: string; user_email?: string }[] } }>(
    page,
    `/team/info?team_id=${encodeURIComponent(teamId)}`,
  );
  return (info.team_info.members_with_roles ?? []).flatMap((member) =>
    [member.user_id, member.user_email].filter((value): value is string => Boolean(value)),
  );
}

/** See keys.spec.ts -- return_full_object is what makes the row carry team_id. */
async function findKeyByAlias(page: PlaywrightPage, alias: string): Promise<Record<string, any> | undefined> {
  const body = await readBack<{ keys: Record<string, any>[] }>(
    page,
    `/key/list?key_alias=${encodeURIComponent(alias)}&return_full_object=true&size=100`,
  );
  return body.keys.find((row) => row.key_alias === alias);
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
    await expect(page.getByText(E2E_INTERNAL_USER_KEY_ALIAS).first()).toBeVisible({ timeout: 10_000 });

    // And from the global Virtual Keys page, the same key should be visible.
    await navigateToPage(page, Page.ApiKeys);
    await expect(page.getByText(E2E_INTERNAL_USER_KEY_ALIAS).first()).toBeVisible({ timeout: 10_000 });
  });

  test("Team admin can add a member to their team", async ({ page }) => {
    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);

    await clickTeamId(page, E2E_TEAM_CRUD_ID);

    await page.getByRole("tab", { name: "Members" }).click();
    await page.getByRole("button", { name: /Add Member/i }).click();

    const modal = page.getByRole("dialog", { name: "Add Team Member" });
    await expect(modal).toBeVisible({ timeout: 5_000 });

    // Use a dedicated invitee user so this doesn't race with the proxy-admin
    // "Invite a user" test that adds invitable@test.local to the same team.
    await modal.getByRole("combobox").first().click();
    await page.keyboard.type("invitable-team@test.local");

    const emailOption = page.getByRole("option", { name: "invitable-team@test.local" }).first();
    await expect(emailOption).toBeAttached({ timeout: 10_000 });
    await page.keyboard.press("Enter");

    const add = await captureRequestBody(page, { method: "POST", urlIncludes: "/team/member_add" }, async () => {
      await modal.getByRole("button", { name: /Add Member/i }).click();
    });
    // An add carrying the wrong team_id still toasts success, and the member lands elsewhere.
    expect(add.team_id, "add targets the team being viewed").toBe(E2E_TEAM_CRUD_ID);
    expect(add.member?.user_email, "the typed email is what goes on the wire").toBe("invitable-team@test.local");

    await expect(page.getByText("Team member added successfully").first()).toBeVisible({ timeout: 10_000 });

    // Membership is the point of the flow, so read the roster back.
    await expect
      .poll(async () => await teamMemberIdentities(page, E2E_TEAM_CRUD_ID), {
        message: "added member never appeared in the team's roster",
        timeout: 15_000,
      })
      .toContain("invitable-team@test.local");
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

    const modal = page.getByRole("dialog", { name: "Delete Team Member" });
    await expect(modal).toBeVisible({ timeout: 5_000 });

    const remove = await captureRequestBody(page, { method: "POST", urlIncludes: "/team/member_delete" }, async () => {
      await modal.getByRole("button", { name: /^Delete$/ }).click();
    });
    // Removing the wrong member is exactly what a success toast hides, so pin both halves.
    expect(remove.team_id, "delete targets the team being viewed").toBe(E2E_TEAM_CRUD_ID);
    expect([remove.user_id, remove.user_email], "delete identifies the member whose row was clicked").toContain(
      "e2e-removable-member",
    );

    await expect(page.getByText("Team member removed successfully").first()).toBeVisible({ timeout: 10_000 });

    // The row disappearing is local state, which happens whether or not the write landed.
    await expect
      .poll(async () => await teamMemberIdentities(page, E2E_TEAM_CRUD_ID), {
        message: "removed member is still on the team",
        timeout: 15_000,
      })
      .not.toContain("e2e-removable-member");
  });

  test("Team admin can create a team key with All Team Models", async ({ page }) => {
    await navigateToPage(page, Page.ApiKeys);
    await dismissFeedbackPopup(page);

    await page.getByRole("button", { name: /Create New Key/i }).click();
    await expect(page.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });

    const keyName = `e2e-team-admin-key-${Date.now()}`;
    await page.getByLabel(/Key Name/).fill(keyName);

    // Team selector — same locator pattern as the proxy-admin keys test.
    const teamSelect = page.getByTestId("team-dropdown").getByRole("combobox");
    await teamSelect.click();
    await page.keyboard.type(E2E_TEAM_CRUD_ALIAS);
    await page.locator('[data-slot="combobox-content"]:visible').getByText(E2E_TEAM_CRUD_ALIAS).first().click();

    // Models — pick "All Team Models". The popup is portaled to the body, so
    // scope the option lookup to the page.
    await page.getByRole("combobox", { name: "Select models" }).click();
    await page.getByRole("option", { name: "All Team Models", exact: true }).click();
    await page.keyboard.press("Escape");

    const generate = await captureRequestBody(page, { method: "POST", urlIncludes: "/key/generate" }, async () => {
      await page.getByRole("button", { name: "Create Key", exact: true }).click();
    });
    expect(generate.team_id, "the selected team goes on the wire").toBe(E2E_TEAM_CRUD_ID);

    await expect(page.getByText("Save your Key")).toBeVisible({ timeout: 10_000 });
    await page.keyboard.press("Escape");

    await expect(page.getByText(keyName)).toBeVisible({ timeout: 10_000 });

    // A team-admin key that comes back unscoped, or scoped elsewhere, is a privilege and
    // billing problem that only a read-back sees.
    const persisted = await findKeyByAlias(page, keyName);
    expect(persisted, `key ${keyName} readable from /key/list`).toBeTruthy();
    expect(persisted?.team_id, "the key is owned by the team admin's own team").toBe(E2E_TEAM_CRUD_ID);
  });
});
