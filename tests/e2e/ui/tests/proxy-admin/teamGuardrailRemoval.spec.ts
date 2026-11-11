import { test, expect, type APIRequestContext, type Page as PlaywrightPage } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage, dismissFeedbackPopup, clickTeamId } from "../../helpers/navigation";
import { proxyIsPremium } from "../../helpers/premium";
import { readBack } from "../../helpers/roundTrip";
import { CHAT_MODEL_A, MOCK_RESPONSE_TEXT, masterKey } from "../../helpers/traffic";

const auth = () => ({ Authorization: `Bearer ${masterKey()}` });

async function guardrailId(request: APIRequestContext, name: string): Promise<string | undefined> {
  const res = await request.get("/v2/guardrails/list", { headers: auth() });
  expect(res.ok(), `GET /v2/guardrails/list (${res.status()})`).toBe(true);
  const rows = (await res.json()).guardrails as { guardrail_id: string; guardrail_name: string | null }[];
  return rows.find((row) => row.guardrail_name === name)?.guardrail_id;
}

async function teamGuardrails(page: PlaywrightPage, teamId: string): Promise<string[]> {
  const body = await readBack<{ team_info: { metadata: { guardrails?: string[] } | null } }>(
    page,
    `/team/info?team_id=${encodeURIComponent(teamId)}`,
  );
  return body.team_info.metadata?.guardrails ?? [];
}

async function keywordPromptStatus(request: APIRequestContext, apiKey: string, keyword: string): Promise<number> {
  const res = await request.post("/v1/chat/completions", {
    headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
    data: { model: CHAT_MODEL_A, messages: [{ role: "user", content: `please tell me about ${keyword}` }] },
  });
  return res.status();
}

test.describe("Proxy Admin - Team guardrail removal", () => {
  test.use({ storageState: ADMIN_STORAGE_PATH });

  test("Clearing a team's only guardrail on the Settings tab lets blocked traffic through again", async ({
    page,
    request,
  }) => {
    test.skip(!proxyIsPremium(), "proxy under test is unlicensed, so team guardrails are premium-gated");

    const stamp = `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
    const guardrailName = `e2e-team-guardrail-${stamp}`;
    const bannedKeyword = `e2eteamban${stamp}`;
    const teamAlias = `e2e-guardrail-team-${stamp}`;

    let teamId = "";
    let teamKey = "";
    try {
      const guardrailRes = await request.post("/guardrails", {
        headers: auth(),
        data: {
          guardrail: {
            guardrail_name: guardrailName,
            litellm_params: {
              guardrail: "litellm_content_filter",
              mode: "pre_call",
              default_on: false,
              blocked_words: [{ keyword: bannedKeyword, action: "BLOCK" }],
            },
          },
        },
      });
      expect(
        guardrailRes.ok(),
        `POST /guardrails failed (${guardrailRes.status()}): ${await guardrailRes.text()}`,
      ).toBe(true);

      const teamRes = await request.post("/team/new", {
        headers: auth(),
        data: { team_alias: teamAlias, models: [CHAT_MODEL_A], metadata: { guardrails: [guardrailName] } },
      });
      expect(teamRes.ok(), `POST /team/new failed (${teamRes.status()}): ${await teamRes.text()}`).toBe(true);
      teamId = (await teamRes.json()).team_id as string;

      const keyRes = await request.post("/key/generate", { headers: auth(), data: { team_id: teamId } });
      expect(keyRes.ok(), `POST /key/generate failed (${keyRes.status()}): ${await keyRes.text()}`).toBe(true);
      teamKey = (await keyRes.json()).key as string;

      await expect
        .poll(async () => keywordPromptStatus(request, teamKey, bannedKeyword), {
          message: "the team's guardrail never started refusing the banned keyword",
          timeout: 60_000,
        })
        .toBe(400);

      await navigateToPage(page, Page.Teams);
      await dismissFeedbackPopup(page);
      await clickTeamId(page, teamId);
      await page.getByRole("tab", { name: "Settings" }).click();
      await page.getByRole("button", { name: "Edit Settings" }).click();

      const chip = page.locator('[data-slot="combobox-chip"]').filter({ hasText: guardrailName });
      await expect(chip).toBeVisible({ timeout: 10_000 });
      await chip.locator('[data-slot="combobox-chip-remove"]').click();
      await expect(chip).toHaveCount(0, { timeout: 10_000 });

      await page.getByRole("button", { name: "Save Changes" }).click();

      await expect
        .poll(async () => teamGuardrails(page, teamId), {
          message: "the team still carries a guardrail in /team/info after the save",
          timeout: 20_000,
        })
        .toEqual([]);

      await page.reload();
      await page.getByRole("tab", { name: "Settings" }).click();
      await page.getByRole("button", { name: "Edit Settings" }).click();
      await expect(page.getByRole("combobox", { name: "Select guardrails" })).toBeVisible({ timeout: 15_000 });
      await expect(
        page.locator('[data-slot="combobox-chip"]').filter({ hasText: guardrailName }),
        "the removed guardrail is gone from the Settings tab after a reload",
      ).toHaveCount(0);

      await expect
        .poll(async () => keywordPromptStatus(request, teamKey, bannedKeyword), {
          message: "the team key is still refused for a keyword whose guardrail was removed",
          timeout: 60_000,
        })
        .toBe(200);

      const served = await request.post("/v1/chat/completions", {
        headers: { Authorization: `Bearer ${teamKey}`, "Content-Type": "application/json" },
        data: {
          model: CHAT_MODEL_A,
          messages: [{ role: "user", content: `please tell me about ${bannedKeyword}` }],
        },
      });
      expect((await served.json()).choices?.[0]?.message?.content).toContain(MOCK_RESPONSE_TEXT);
    } finally {
      if (teamKey) {
        await request.post("/key/delete", { headers: auth(), data: { keys: [teamKey] } });
      }
      if (teamId) {
        await request.post("/team/delete", { headers: auth(), data: { team_ids: [teamId] } });
      }
      const id = await guardrailId(request, guardrailName);
      if (id) {
        await request.delete(`/guardrails/${id}`, { headers: auth() });
      }
    }
  });
});
