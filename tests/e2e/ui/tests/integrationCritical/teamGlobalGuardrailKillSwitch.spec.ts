import {
  test,
  expect,
  APIRequestContext,
  Page as PlaywrightPage,
} from "@playwright/test";
import { randomUUID } from "node:crypto";

const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
const headers = { Authorization: `Bearer ${master}` };

async function createTeam(request: APIRequestContext): Promise<string> {
  const created = await request.post("/team/new", {
    headers,
    data: {
      team_alias: `int_kill_switch_${randomUUID().replace(/-/g, "").slice(0, 12)}`,
    },
  });
  expect(created.ok(), await created.text()).toBe(true);
  return (await created.json()).team_id as string;
}

async function loginAsAdmin(page: PlaywrightPage): Promise<void> {
  await page.goto("/ui/login");
  await page.getByPlaceholder("Enter your username").fill("admin");
  await page.getByPlaceholder("Enter your password").fill(master);
  await page.getByRole("button", { name: "Login", exact: true }).click();
  await expect(page).toHaveURL(
    (url) => url.pathname.startsWith("/ui") && !url.pathname.includes("login"),
  );
}

test("proxy admin can enable the global guardrail kill switch from the models page team drill-in", async ({
  page,
  request,
}) => {
  const teamId = await createTeam(request);
  try {
    await loginAsAdmin(page);
    await page.goto(`/ui/models-and-endpoints?team=${teamId}`);
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: /edit settings/i }).click();
    await expect(page.getByLabel(/Team Name/)).toBeVisible();
    const killSwitch = page.getByRole("switch", {
      name: /disable all global guardrails/i,
    });
    await expect(killSwitch).toBeVisible();
    await expect(killSwitch).not.toBeChecked();
    await killSwitch.click();
    await page.getByRole("button", { name: "Save Changes" }).click();
    await expect
      .poll(async () => {
        const response = await request.get(`/team/info?team_id=${teamId}`, {
          headers,
        });
        expect(response.ok(), await response.text()).toBe(true);
        const json = await response.json();
        return json.team_info?.metadata?.disable_global_guardrails;
      })
      .toBe(true);
    await page.reload();
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: /edit settings/i }).click();
    await expect(page.getByLabel(/Team Name/)).toBeVisible();
    await expect(
      page.getByRole("switch", { name: /disable all global guardrails/i }),
    ).toBeChecked();
  } finally {
    const removed = await request.post("/team/delete", {
      headers,
      data: { team_ids: [teamId] },
    });
    expect(removed.ok() || removed.status() === 404, await removed.text()).toBe(
      true,
    );
  }
});
