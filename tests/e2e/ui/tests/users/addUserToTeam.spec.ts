import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { masterKey } from "../../helpers/traffic";

const auth = () => ({ Authorization: `Bearer ${masterKey()}` });

async function teamRoleOf(page: PlaywrightPage, teamId: string, userId: string): Promise<string | undefined> {
  const info = await readBack<{ team_info: { members_with_roles: { user_id?: string; role?: string }[] } }>(
    page,
    `/team/info?team_id=${encodeURIComponent(teamId)}`,
  );
  return info.team_info.members_with_roles.find((member) => member.user_id === userId)?.role;
}

test.describe("Internal Users - Add to team", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Adding a user to a team from their user page makes them a member with the chosen role", async ({
    page,
    request,
  }) => {
    const stamp = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
    const userId = `e2e-add-to-team-${stamp}`;
    const email = `${userId}@test.local`;
    const teamAlias = `e2e-add-to-team-${stamp}`;
    let teamId = "";
    try {
      const createdUser = await request.post("/user/new", {
        headers: auth(),
        data: { user_id: userId, user_email: email, user_role: "internal_user", auto_create_key: false },
      });
      expect(createdUser.ok(), `POST /user/new (${createdUser.status()}): ${await createdUser.text()}`).toBe(true);
      const createdTeam = await request.post("/team/new", { headers: auth(), data: { team_alias: teamAlias } });
      expect(createdTeam.ok(), `POST /team/new (${createdTeam.status()}): ${await createdTeam.text()}`).toBe(true);
      teamId = (await createdTeam.json()).team_id as string;

      await navigateToPage(page, Page.Users);
      await page.getByPlaceholder("Search by email or ID…").fill(email);
      await page.getByText(userId, { exact: true }).first().click();
      await page.getByRole("button", { name: "Add Team" }).click();
      const dialog = page.getByRole("dialog");
      await expect(dialog).toBeVisible();

      const teamPicker = dialog.getByRole("combobox").first();
      await teamPicker.click();
      await teamPicker.pressSequentially(teamAlias);
      await page.getByRole("option", { name: teamAlias }).click();
      await dialog.getByLabel("Member Role").click();
      await page.getByRole("option", { name: "admin" }).click();
      await dialog.getByRole("button", { name: "Add to Team" }).click();
      await expect(dialog).toBeHidden();

      await expect
        .poll(async () => teamRoleOf(page, teamId, userId), {
          message: "user never appeared in the team with the chosen role",
          timeout: 15_000,
        })
        .toBe("admin");
    } finally {
      if (teamId) {
        await request.post("/team/delete", { headers: auth(), data: { team_ids: [teamId] } });
      }
      await request.post("/user/delete", { headers: auth(), data: { user_ids: [userId] } });
    }
  });
});
