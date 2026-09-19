import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup, clickTeamId } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { CHAT_MODEL_A, masterKey } from "../../helpers/traffic";

interface TeamInfoResponse {
  team_info: {
    models: string[];
    members_with_roles: { user_id?: string; role?: string }[];
  };
  team_memberships: {
    user_id: string;
    litellm_budget_table: { max_budget: number | null } | null;
  }[];
}

const auth = () => ({ Authorization: `Bearer ${masterKey()}` });

async function teamInfo(page: PlaywrightPage, teamId: string): Promise<TeamInfoResponse> {
  return readBack<TeamInfoResponse>(page, `/team/info?team_id=${encodeURIComponent(teamId)}`);
}

function roleOf(info: TeamInfoResponse, userId: string): string | undefined {
  return info.team_info.members_with_roles.find((member) => member.user_id === userId)?.role;
}

function budgetOf(info: TeamInfoResponse, userId: string): number | null | undefined {
  return info.team_memberships.find((membership) => membership.user_id === userId)?.litellm_budget_table?.max_budget;
}

function otherMembers(info: TeamInfoResponse, userId: string): string[] {
  return info.team_info.members_with_roles
    .filter((member) => member.user_id !== userId)
    .map((member) => `${member.user_id}:${member.role}`)
    .sort();
}

test.describe("Proxy Admin - Team member edit", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  const createdTeams: string[] = [];
  const createdUsers: string[] = [];

  test.afterEach(async ({ request }) => {
    for (const teamId of createdTeams.splice(0)) {
      await request.post("/team/delete", { headers: auth(), data: { team_ids: [teamId] } });
    }
    for (const userId of createdUsers.splice(0)) {
      await request.post("/user/delete", { headers: auth(), data: { user_ids: [userId] } });
    }
  });

  test("Editing a member's role and per-member budget persists and survives a reload", async ({ page, request }) => {
    const stamp = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
    const memberId = `e2e-member-edit-${stamp}`;
    const teamAlias = `e2e-member-edit-team-${stamp}`;

    createdUsers.push(memberId);
    const created = await request.post("/user/new", {
      headers: auth(),
      data: { user_id: memberId, user_role: "internal_user", auto_create_key: false },
    });
    expect(created.ok(), `POST /user/new failed (${created.status()}): ${await created.text()}`).toBe(true);

    const teamRes = await request.post("/team/new", {
      headers: auth(),
      data: {
        team_alias: teamAlias,
        models: [CHAT_MODEL_A],
        members_with_roles: [{ user_id: memberId, role: "admin" }],
      },
    });
    expect(teamRes.ok(), `POST /team/new failed (${teamRes.status()}): ${await teamRes.text()}`).toBe(true);
    const teamId = (await teamRes.json()).team_id as string;
    createdTeams.push(teamId);

    const before = await teamInfo(page, teamId);
    expect(roleOf(before, memberId), "the member starts out as a team admin").toBe("admin");
    expect(budgetOf(before, memberId) ?? null, "the member starts out with no per-member budget").toBeNull();
    expect(
      otherMembers(before, memberId).length,
      "the team has another member for the edit to leave alone",
    ).toBeGreaterThan(0);

    await navigateToPage(page, Page.Teams);
    await dismissFeedbackPopup(page);
    await clickTeamId(page, teamId);
    await page.getByRole("tab", { name: "Members" }).click();

    const memberRow = page.locator("tr", { hasText: memberId }).first();
    await expect(memberRow).toBeVisible({ timeout: 10_000 });
    await memberRow.getByTestId("edit-member").click();

    const modal = page.getByRole("dialog", { name: "Edit Member" });
    await expect(modal).toBeVisible({ timeout: 10_000 });

    await modal.getByLabel(/^Role/).click();
    await page.getByRole("option", { name: "User", exact: true }).click();
    await modal.getByLabel(/Team Member Budget \(USD\)/).fill("5");
    await modal.getByRole("button", { name: "Save Changes" }).click();

    await expect(page.getByText("Team member updated successfully").first()).toBeVisible({ timeout: 10_000 });

    await expect
      .poll(
        async () => {
          const info = await teamInfo(page, teamId);
          return [roleOf(info, memberId), budgetOf(info, memberId)];
        },
        { message: "the member's role and budget never landed in /team/info", timeout: 20_000 },
      )
      .toEqual(["user", 5]);

    await page.reload();
    await page.getByRole("tab", { name: "Members" }).click();
    const reloadedRow = page.locator("tr", { hasText: memberId }).first();
    await expect(reloadedRow).toBeVisible({ timeout: 15_000 });
    await expect(reloadedRow.getByText("user", { exact: true }), "role shown after a reload").toBeVisible();
    await expect(reloadedRow.getByText("$5.00"), "per-member budget shown after a reload").toBeVisible();

    const after = await teamInfo(page, teamId);
    expect(after.team_info.models, "model access untouched by a member edit").toEqual(before.team_info.models);
    expect(otherMembers(after, memberId), "the rest of the roster untouched by a member edit").toEqual(
      otherMembers(before, memberId),
    );
  });
});
