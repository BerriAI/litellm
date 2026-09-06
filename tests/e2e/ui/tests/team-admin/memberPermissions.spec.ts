import { test, expect, type Browser, type BrowserContext, type Page as PlaywrightPage } from "@playwright/test";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup, clickTeamId } from "../../helpers/navigation";
import { CHAT_MODEL_A, MOCK_RESPONSE_TEXT, masterKey } from "../../helpers/traffic";

const PASSWORD = "E2e-Member-Perms-Pass-1!";

const auth = () => ({ Authorization: `Bearer ${masterKey()}` });

async function sessionKey(page: PlaywrightPage): Promise<string> {
  const cookie = (await page.context().cookies()).find((candidate) => candidate.name === "token");
  expect(cookie?.value, "logged-in session carries a token cookie").toBeTruthy();
  const payload = JSON.parse(Buffer.from(cookie!.value.split(".")[1], "base64url").toString("utf-8")) as {
    key?: string;
  };
  expect(payload.key, "session JWT carries the virtual key the dashboard calls with").toMatch(/^sk-/);
  return payload.key!;
}

async function signIn(browser: Browser, email: string): Promise<BrowserContext> {
  const context = await browser.newContext({ storageState: { cookies: [], origins: [] } });
  const page = await context.newPage();
  await page.goto("/ui/login");
  await page.getByPlaceholder("Enter your username").fill(email);
  await page.getByPlaceholder("Enter your password").fill(PASSWORD);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await expect(page.locator("a", { hasText: "Virtual Keys" })).toBeVisible({ timeout: 30_000 });
  await dismissFeedbackPopup(page);
  return context;
}

test.describe("Team Admin - Member permissions", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("Granting /key/generate lets a plain member create a team key that serves traffic", async ({
    browser,
    request,
  }) => {
    const stamp = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
    const adminId = `e2e-perm-admin-${stamp}`;
    const memberId = `e2e-perm-member-${stamp}`;
    const adminEmail = `${adminId}@test.local`;
    const memberEmail = `${memberId}@test.local`;
    const teamAlias = `e2e-perm-team-${stamp}`;

    const createUser = async (userId: string, email: string): Promise<void> => {
      const created = await request.post("/user/new", {
        headers: auth(),
        data: { user_id: userId, user_email: email, user_role: "internal_user", auto_create_key: false },
      });
      expect(created.ok(), `POST /user/new for ${userId} (${created.status()}): ${await created.text()}`).toBe(true);
      const password = await request.post("/user/update", {
        headers: auth(),
        data: { user_id: userId, password: PASSWORD },
      });
      expect(password.ok(), `POST /user/update for ${userId} (${password.status()})`).toBe(true);
    };

    let teamId = "";
    const createdKeys: string[] = [];
    const contexts: BrowserContext[] = [];
    try {
      await createUser(adminId, adminEmail);
      await createUser(memberId, memberEmail);

      const teamRes = await request.post("/team/new", {
        headers: auth(),
        data: {
          team_alias: teamAlias,
          models: [CHAT_MODEL_A],
          members_with_roles: [
            { user_id: adminId, role: "admin" },
            { user_id: memberId, role: "user" },
          ],
        },
      });
      expect(teamRes.ok(), `POST /team/new failed (${teamRes.status()}): ${await teamRes.text()}`).toBe(true);
      teamId = (await teamRes.json()).team_id as string;

      const memberContext = await signIn(browser, memberEmail);
      contexts.push(memberContext);
      const memberPage = memberContext.pages()[0];
      const memberSessionKey = await sessionKey(memberPage);

      const refused = await memberPage.request.post("/key/generate", {
        headers: { Authorization: `Bearer ${memberSessionKey}`, "Content-Type": "application/json" },
        data: { team_id: teamId, key_alias: `e2e-perm-denied-${stamp}` },
      });
      expect(refused.status(), "a plain member cannot mint a team key before the grant").toBe(401);
      expect(await refused.text()).toContain("/key/generate");

      const adminContext = await signIn(browser, adminEmail);
      contexts.push(adminContext);
      const adminPage = adminContext.pages()[0];
      await navigateToPage(adminPage, Page.Teams);
      await clickTeamId(adminPage, teamId);
      await adminPage.getByRole("tab", { name: "Member Permissions" }).click();

      for (const route of ["/key/generate", "/key/update"]) {
        await adminPage.getByRole("row").filter({ hasText: route }).getByRole("checkbox").check();
      }
      await adminPage.getByRole("button", { name: "Save Changes" }).click();
      await expect(adminPage.getByText("Permissions updated successfully").first()).toBeVisible({ timeout: 10_000 });

      await expect
        .poll(
          async () => {
            const res = await request.get(`/team/permissions_list?team_id=${encodeURIComponent(teamId)}`, {
              headers: auth(),
            });
            if (!res.ok()) return [];
            return ((await res.json()).team_member_permissions ?? []) as string[];
          },
          { message: "the granted permissions never landed in /team/permissions_list", timeout: 20_000 },
        )
        .toEqual(expect.arrayContaining(["/key/generate", "/key/update"]));

      const keyAlias = `e2e-perm-key-${stamp}`;
      await navigateToPage(memberPage, Page.ApiKeys);
      await memberPage.getByRole("button", { name: /Create New Key/i }).click();
      await expect(memberPage.getByText("Key Ownership")).toBeVisible({ timeout: 10_000 });
      await memberPage.getByLabel(/Key Name/).fill(keyAlias);

      const teamSelect = memberPage.getByTestId("team-dropdown").getByRole("combobox");
      await teamSelect.click();
      await memberPage.keyboard.type(teamAlias);
      await memberPage.getByRole("option", { name: teamAlias }).first().click();

      await memberPage.getByRole("combobox", { name: "Select models" }).click();
      await memberPage.getByRole("option", { name: "All Team Models", exact: true }).click();
      await memberPage.keyboard.press("Escape");

      await memberPage.getByRole("button", { name: "Create Key", exact: true }).click();
      const saveDialog = memberPage.getByRole("dialog", { name: "Save your Key" });
      await expect(saveDialog).toBeVisible({ timeout: 15_000 });
      const apiKey = (await saveDialog.locator("pre").innerText()).trim();
      expect(apiKey).toMatch(/^sk-/);
      createdKeys.push(apiKey);
      await memberPage.keyboard.press("Escape");

      await expect
        .poll(
          async () => {
            const res = await request.get(
              `/key/list?team_id=${encodeURIComponent(teamId)}&return_full_object=true&size=100`,
              { headers: auth() },
            );
            if (!res.ok()) return null;
            const row = ((await res.json()).keys as Record<string, unknown>[]).find(
              (candidate) => candidate.key_alias === keyAlias,
            );
            return row ? [row.user_id, row.team_id] : null;
          },
          { message: `key ${keyAlias} never appeared on the team with the member as its owner`, timeout: 20_000 },
        )
        .toEqual([memberId, teamId]);

      const served = await request.post("/v1/chat/completions", {
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        data: { model: CHAT_MODEL_A, messages: [{ role: "user", content: `member key ping ${stamp}` }] },
      });
      expect(served.status(), "the delegated key is a real key the gateway serves").toBe(200);
      expect((await served.json()).choices?.[0]?.message?.content).toContain(MOCK_RESPONSE_TEXT);
    } finally {
      for (const context of contexts) {
        await context.close();
      }
      for (const key of createdKeys) {
        await request.post("/key/delete", { headers: auth(), data: { keys: [key] } });
      }
      if (teamId) {
        await request.post("/team/delete", { headers: auth(), data: { team_ids: [teamId] } });
      }
      await request.post("/user/delete", { headers: auth(), data: { user_ids: [adminId, memberId] } });
    }
  });
});
