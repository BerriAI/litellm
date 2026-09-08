import { test, expect, type APIRequestContext } from "@playwright/test";
import { Page } from "../../fixtures/pages";
import {
  dismissFeedbackPopup,
  navigateToPage,
  openKeyDetail,
} from "../../helpers/navigation";
import {
  CHAT_MODEL_A,
  CHAT_MODEL_B,
  MOCK_RESPONSE_TEXT,
  attemptChatCompletion,
  createVirtualKey,
  deleteVirtualKey,
  masterKey,
  readKeyInfo,
  rootPath,
  uniqueSuffix,
} from "../../helpers/traffic";

const MEMBER_PASSWORD = "E2e-Team-Member-Pass-1!";

interface CreatedTeam {
  readonly team_id: string;
}

function assertCreatedTeam(body: unknown): asserts body is CreatedTeam {
  expect(body, "/team/new returned no team_id").toMatchObject({
    team_id: expect.any(String),
  });
}

async function postAsMaster(
  request: APIRequestContext,
  path: string,
  data: Record<string, unknown>,
): Promise<unknown> {
  const res = await request.post(`${rootPath()}${path}`, {
    headers: {
      Authorization: `Bearer ${masterKey()}`,
      "Content-Type": "application/json",
    },
    data,
  });
  expect(
    res.ok(),
    `POST ${path} failed (${res.status()}): ${await res.text()}`,
  ).toBe(true);
  return res.json();
}

test.describe("Internal User - own team key model scope", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("a team member narrows their own key's models and the proxy enforces it", async ({
    page,
    request,
  }) => {
    const suffix = uniqueSuffix();
    const email = `team-member-${suffix}@test.local`;
    const userId = `e2e-key-scope-user-${suffix}`;
    const alias = `e2e-key-scope-${suffix}`;

    const team = await postAsMaster(request, "/team/new", {
      team_alias: `E2E Key Scope ${suffix}`,
      models: [CHAT_MODEL_A, CHAT_MODEL_B],
      team_member_permissions: ["/key/generate", "/key/update", "/key/info"],
    });
    assertCreatedTeam(team);
    const teamId = team.team_id;

    try {
      await postAsMaster(request, "/user/new", {
        user_id: userId,
        user_email: email,
        user_role: "internal_user",
        auto_create_key: false,
      });
      await postAsMaster(request, "/user/update", {
        user_id: userId,
        password: MEMBER_PASSWORD,
      });
      await postAsMaster(request, "/team/member_add", {
        team_id: teamId,
        member: { role: "user", user_id: userId },
      });

      const created = await createVirtualKey(request, {
        key_alias: alias,
        team_id: teamId,
        user_id: userId,
        models: [],
      });

      try {
        await page.goto("/ui/login");
        await page.getByPlaceholder("Enter your username").fill(email);
        await page
          .getByPlaceholder("Enter your password")
          .fill(MEMBER_PASSWORD);
        await page.getByRole("button", { name: "Login", exact: true }).click();
        await expect(
          page.locator("a", { hasText: "Virtual Keys" }),
          `${email} never reached the dashboard`,
        ).toBeVisible({ timeout: 30_000 });
        await dismissFeedbackPopup(page);

        await navigateToPage(page, Page.ApiKeys);
        await openKeyDetail(page, alias);

        await page.getByRole("tab", { name: "Settings" }).click();
        await page.getByRole("button", { name: "Edit Settings" }).click();

        await page.getByRole("combobox", { name: "Select models" }).click();
        await expect(
          page.getByRole("option", { name: CHAT_MODEL_A, exact: true }),
          `the Models dropdown does not offer ${CHAT_MODEL_A} to a team member`,
        ).toBeVisible({ timeout: 15_000 });
        await expect(
          page.getByRole("option", { name: CHAT_MODEL_B, exact: true }),
          `the Models dropdown does not offer ${CHAT_MODEL_B} to a team member`,
        ).toBeVisible();

        await page
          .getByRole("option", { name: CHAT_MODEL_A, exact: true })
          .click();
        await page.keyboard.press("Escape");

        const updated = page.waitForResponse(
          (res) =>
            res.url().includes("/key/update") &&
            res.request().method() === "POST",
        );
        await page.getByRole("button", { name: "Save Changes" }).click();
        const updateStatus = (await updated).status();
        expect(
          updateStatus,
          "a team member's own-key edit was refused",
        ).toBeGreaterThanOrEqual(200);
        expect(
          updateStatus,
          "a team member's own-key edit was refused",
        ).toBeLessThan(300);
        await expect(
          page.getByText("Key updated successfully").first(),
        ).toBeVisible({ timeout: 15_000 });

        await expect
          .poll(
            async () => (await readKeyInfo(request, created.token)).models,
            {
              message: `the narrowed model scope never reached /key/info for ${alias}`,
              timeout: 20_000,
            },
          )
          .toEqual([CHAT_MODEL_A]);

        await expect
          .poll(
            async () =>
              await attemptChatCompletion(request, {
                model: CHAT_MODEL_B,
                prompt: `out of scope ${suffix}`,
                apiKey: created.key,
              }),
            {
              message: `${CHAT_MODEL_B} was still served after the key was narrowed to ${CHAT_MODEL_A}`,
              timeout: 30_000,
            },
          )
          .toMatchObject({
            status: 403,
            body: expect.stringContaining(CHAT_MODEL_B),
          });

        const inScope = await attemptChatCompletion(request, {
          model: CHAT_MODEL_A,
          prompt: `in scope ${suffix}`,
          apiKey: created.key,
        });
        expect(
          inScope,
          `${CHAT_MODEL_A} is no longer served by the narrowed key`,
        ).toMatchObject({
          status: 200,
          body: expect.stringContaining(MOCK_RESPONSE_TEXT),
        });
      } finally {
        await deleteVirtualKey(request, created.token);
      }
    } finally {
      await request.post(`${rootPath()}/user/delete`, {
        headers: {
          Authorization: `Bearer ${masterKey()}`,
          "Content-Type": "application/json",
        },
        data: { user_ids: [userId] },
      });
      await request.post(`${rootPath()}/team/delete`, {
        headers: {
          Authorization: `Bearer ${masterKey()}`,
          "Content-Type": "application/json",
        },
        data: { team_ids: [teamId] },
      });
    }
  });
});
