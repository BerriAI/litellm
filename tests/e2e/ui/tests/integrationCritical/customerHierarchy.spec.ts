import { test as base, expect, type APIRequestContext, type Page } from "@playwright/test";
import { dismissFeedbackPopup } from "../../helpers/navigation";
import { masterKey, uniqueSuffix } from "../../helpers/traffic";

const api = "/management/v1";
const headers = () => ({ Authorization: `Bearer ${masterKey()}` });

const test = base.extend<{ organizationId: string }>({
  organizationId: async ({ request }, use) => {
    const id = `e2e-customer-hierarchy-${uniqueSuffix()}`;
    const created = await request.post(`${api}/customer_organizations`, {
      headers: headers(),
      data: { customer_org_id: id, alias: id },
    });
    expect(created.ok(), await created.text()).toBe(true);
    try {
      await use(id);
    } finally {
      const removed = await request.delete(`${api}/customer_organizations/${id}?cascade=true`, { headers: headers() });
      expect(removed.ok(), await removed.text()).toBe(true);
    }
  },
});

async function openMembers(page: Page, organizationId: string) {
  await page.goto(`/ui/customers/manage?${new URLSearchParams({ org: organizationId, tab: "members" })}`);
  await expect(page.getByRole("button", { name: "Create customer member" })).toBeVisible();
  await dismissFeedbackPopup(page);
}

async function getMember(request: APIRequestContext, memberId: string) {
  const member = await request.get(`${api}/customer_members/${memberId}`, { headers: headers() });
  expect(member.ok(), await member.text()).toBe(true);
  return member.json();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/ui/login");
  await page.getByPlaceholder("Enter your username").fill("admin");
  await page.getByPlaceholder("Enter your password").fill(masterKey());
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await expect(page).toHaveURL((url) => url.pathname.startsWith("/ui") && !url.pathname.includes("login"));
});

test("creating an organization member without a team creates and assigns the default team", async ({
  page,
  request,
  organizationId,
}) => {
  const defaultTeamId = `${organizationId}:default`;
  const memberId = `${organizationId}-member`;
  const teamPath = `${api}/customer_teams/${encodeURIComponent(defaultTeamId)}`;
  const before = await request.get(teamPath, { headers: headers() });
  expect(before.status()).toBe(404);

  await openMembers(page, organizationId);
  await page.getByRole("button", { name: "Create customer member" }).click();
  const dialog = page.getByRole("dialog", { name: "Create customer member" });
  await dialog.getByLabel("Custom identifier (optional)").fill(memberId);
  await dialog.getByLabel("Display name").fill("Member with default team");
  await dialog.getByRole("button", { name: "Create", exact: true }).click();
  await expect(dialog).toBeHidden();

  const team = await request.get(teamPath, { headers: headers() });
  expect(team.ok(), await team.text()).toBe(true);
  expect((await team.json()).parent_org_id).toBe(organizationId);
  expect((await getMember(request, memberId)).parent_team_ids).toEqual([defaultTeamId]);
});

test("moving a member stores the selected default team first", async ({ page, request, organizationId }) => {
  const alphaId = `${organizationId}-alpha`;
  const betaId = `${organizationId}-beta`;
  const memberId = `${organizationId}-member`;
  for (const [id, alias] of [[alphaId, "Alpha team"], [betaId, "Beta team"]]) {
    const team = await request.post(`${api}/customer_teams`, {
      headers: headers(),
      data: { customer_team_id: id, parent_org_id: organizationId, alias },
    });
    expect(team.ok(), await team.text()).toBe(true);
  }
  const member = await request.post(`${api}/customer_members`, {
    headers: headers(),
    data: {
      customer_member_id: memberId,
      parent_org_id: organizationId,
      parent_team_ids: [alphaId],
      alias: "Member to move",
      budget: { max_budget: 11, budget_duration: "1mo" },
    },
  });
  expect(member.ok(), await member.text()).toBe(true);

  await openMembers(page, organizationId);
  await page.getByRole("button", { name: "Actions for Member to move" }).click();
  await page.getByRole("menuitem", { name: "Move teams" }).click();
  const dialog = page.getByRole("dialog", { name: "Move teams" });
  await dialog.getByRole("combobox", { name: "Choose teams" }).click();
  await page.getByRole("option", { name: /Beta team/ }).click();
  await page.keyboard.press("Escape");
  await dialog.getByLabel("Default team").click();
  await page.getByRole("option", { name: "Beta team", exact: true }).click();
  await dialog.getByRole("button", { name: "Move teams", exact: true }).click();
  await expect(dialog).toBeHidden();

  expect((await getMember(request, memberId)).parent_team_ids).toEqual([betaId, alphaId]);
});
