import { test, expect, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup, clickTeamId } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { CHAT_MODEL_A, MOCK_RESPONSE_TEXT, masterKey } from "../../helpers/traffic";

interface TeamInfo {
  team_id: string;
  team_alias: string;
  models: string[];
  max_budget: number | null;
  tpm_limit: number | null;
  rpm_limit: number | null;
  metadata: Record<string, unknown> | null;
  members_with_roles: { user_id?: string; role?: string }[];
}

/**
 * Each test owns a team it created, rather than editing a seeded one, so a save that clobbers a
 * field cannot take another spec's fixture down with it.
 */
async function createTeam(page: PlaywrightPage, alias: string, members: string[] = []): Promise<string> {
  const res = await page.request.post("/team/new", {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: {
      team_alias: alias,
      models: [CHAT_MODEL_A],
      members_with_roles: members.map((user_id) => ({ user_id, role: "user" })),
    },
  });
  expect(res.ok(), `POST /team/new failed (${res.status()}): ${await res.text()}`).toBe(true);
  return (await res.json()).team_id as string;
}

/**
 * A member of this test's own, not one of the seeded users. Putting a seeded user on an extra team
 * changes what every spec that asserts on their memberships sees.
 */
async function createMember(page: PlaywrightPage, userId: string): Promise<string> {
  const res = await page.request.post("/user/new", {
    headers: { Authorization: `Bearer ${masterKey()}` },
    data: { user_id: userId, user_role: "internal_user", auto_create_key: false },
  });
  expect(res.ok(), `POST /user/new failed (${res.status()}): ${await res.text()}`).toBe(true);
  return userId;
}

async function teamInfo(page: PlaywrightPage, teamId: string): Promise<TeamInfo> {
  const body = await readBack<{ team_info: TeamInfo }>(page, `/team/info?team_id=${encodeURIComponent(teamId)}`);
  return body.team_info;
}

async function openTeamSettings(page: PlaywrightPage, teamId: string): Promise<void> {
  await navigateToPage(page, Page.Teams);
  await dismissFeedbackPopup(page);
  await clickTeamId(page, teamId);
  await page.getByRole("tab", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Edit Settings" }).click();
  await expect(page.getByRole("button", { name: "Save Changes" })).toBeVisible({ timeout: 10_000 });
}

test.describe("Proxy Admin - Team settings", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Setting a team's spend cap and rate limits leaves its models and members alone", async ({ page }) => {
    const stamp = Date.now();
    const alias = `e2e-team-limits-${stamp}`;
    const member = await createMember(page, `e2e-team-limits-member-${stamp}`);
    const teamId = await createTeam(page, alias, [member]);
    const before = await teamInfo(page, teamId);

    await openTeamSettings(page, teamId);

    await page.getByRole("spinbutton", { name: "Max Budget (USD)" }).fill("42.5");
    await page.getByRole("spinbutton", { name: "Tokens per minute Limit (TPM)" }).fill("7000");
    await page.getByRole("spinbutton", { name: "Requests per minute Limit (RPM)" }).fill("70");
    await page.getByRole("button", { name: "Save Changes" }).click();

    await expect
      .poll(
        async () => {
          const team = await teamInfo(page, teamId);
          return [team.max_budget, team.tpm_limit, team.rpm_limit];
        },
        { message: "team limits did not persist", timeout: 20_000 },
      )
      .toEqual([42.5, 7000, 70]);

    // The Settings form posts the whole team. A field it fails to seed goes back as null, and
    // the toast still says success, so pin the fields this edit had no business touching.
    const after = await teamInfo(page, teamId);
    expect(after.models, "model access untouched by a limits edit").toEqual(before.models);
    expect(
      after.members_with_roles.map((member) => member.user_id).sort(),
      "membership untouched by a limits edit",
    ).toEqual(before.members_with_roles.map((member) => member.user_id).sort());
  });

  test("A model alias added on the Settings tab serves traffic under the alias name", async ({ page }) => {
    const stamp = Date.now();
    const alias = `e2e-team-alias-${stamp}`;
    const modelAlias = `e2e-alias-${stamp}`;
    const teamId = await createTeam(page, alias);

    await openTeamSettings(page, teamId);

    await page.getByRole("textbox", { name: "Alias Name" }).fill(modelAlias);
    await page.getByRole("combobox", { name: "Select target model" }).click();
    await page.getByRole("option", { name: CHAT_MODEL_A, exact: true }).first().click();
    await page.getByRole("button", { name: "Add Alias" }).click();
    await page.getByRole("button", { name: "Save Changes" }).click();

    await expect
      .poll(async () => (await teamInfo(page, teamId)).models, { message: "team lost its models", timeout: 20_000 })
      .toEqual([CHAT_MODEL_A]);

    const keyRes = await page.request.post("/key/generate", {
      headers: { Authorization: `Bearer ${masterKey()}` },
      data: { team_id: teamId, key_alias: `e2e-alias-key-${stamp}` },
    });
    expect(keyRes.ok(), `POST /key/generate failed (${keyRes.status()})`).toBe(true);
    const teamKey = (await keyRes.json()).key as string;

    // An alias the team can see but cannot call is the actual complaint; the readback alone
    // would pass for an alias the router never resolves.
    const served = await page.request.post("/v1/chat/completions", {
      headers: { Authorization: `Bearer ${teamKey}`, "Content-Type": "application/json" },
      data: { model: modelAlias, messages: [{ role: "user", content: "ping" }] },
    });
    expect(served.status(), `a team key calling ${modelAlias} is served`).toBe(200);
    expect((await served.json()).choices?.[0]?.message?.content).toContain(MOCK_RESPONSE_TEXT);
  });

  test("Team metadata added as key-value pairs survives a reload", async ({ page }) => {
    const stamp = Date.now();
    const alias = `e2e-team-metadata-${stamp}`;
    const metadataValue = `cost-center-${stamp}`;
    const teamId = await createTeam(page, alias);

    await openTeamSettings(page, teamId);

    await page.getByRole("button", { name: "Add Key-Value Pair" }).click();
    await page.getByPlaceholder("Key", { exact: true }).last().fill("owner");
    await page.getByPlaceholder("Value", { exact: true }).last().fill(metadataValue);
    await page.getByRole("button", { name: "Save Changes" }).click();

    await expect
      .poll(async () => (await teamInfo(page, teamId)).metadata?.owner, {
        message: "team metadata did not persist",
        timeout: 20_000,
      })
      .toBe(metadataValue);

    // Reopening the form is the step that catches metadata the page writes but cannot read back.
    await page.reload();
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: "Edit Settings" }).click();
    await expect(page.getByPlaceholder("Key", { exact: true })).toHaveValue("owner", { timeout: 15_000 });
    await expect(page.getByPlaceholder("Value", { exact: true })).toHaveValue(metadataValue);
  });
});
