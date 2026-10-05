import {
  test,
  expect,
  APIRequestContext,
  Page as PlaywrightPage,
} from "@playwright/test";
import { randomUUID } from "node:crypto";

const master = process.env.LITELLM_MASTER_KEY ?? "sk-integration-master";
const headers = { Authorization: `Bearer ${master}` };

async function createTeamCarryingAnEmptyMetadataKey(
  request: APIRequestContext,
): Promise<string> {
  const created = await request.post("/team/new", {
    headers,
    data: {
      team_alias: `int_empty_key_${randomUUID().replace(/-/g, "").slice(0, 12)}`,
    },
  });
  expect(created.ok(), await created.text()).toBe(true);
  const teamId = (await created.json()).team_id as string;
  const seeded = await request.post("/team/update", {
    headers,
    data: {
      team_id: teamId,
      metadata: { "": { displayName: "stale" }, env: "staging" },
    },
  });
  expect(seeded.ok(), await seeded.text()).toBe(true);
  return teamId;
}

async function teamMetadata(
  request: APIRequestContext,
  teamId: string,
): Promise<Record<string, unknown>> {
  const response = await request.get(`/team/info?team_id=${teamId}`, {
    headers,
  });
  expect(response.ok(), await response.text()).toBe(true);
  const json = await response.json();
  return (json.team_info?.metadata ?? {}) as Record<string, unknown>;
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

test("the team settings form skips a metadata row with an empty key and saving drops the key", async ({
  page,
  request,
}) => {
  const teamId = await createTeamCarryingAnEmptyMetadataKey(request);
  try {
    expect(Object.keys(await teamMetadata(request, teamId))).toContain("");
    await loginAsAdmin(page);
    await page.goto(`/ui/models-and-endpoints?team=${teamId}`);
    await page.getByRole("tab", { name: "Settings" }).click();
    await page.getByRole("button", { name: /edit settings/i }).click();
    await expect(page.getByLabel(/Team Name/)).toBeVisible();
    const keys = page.getByPlaceholder("Key", { exact: true });
    await expect(keys).toHaveCount(1);
    await expect(keys.first()).toHaveValue("env");
    await expect(
      page.getByPlaceholder("Value", { exact: true }).first(),
    ).toHaveValue("staging");
    await page.getByRole("button", { name: "Save Changes" }).click();
    await expect
      .poll(async () => {
        const metadata = await teamMetadata(request, teamId);
        return { hasEmptyKey: "" in metadata, env: metadata.env };
      })
      .toEqual({ hasEmptyKey: false, env: "staging" });
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
