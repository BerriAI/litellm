import { randomUUID } from "node:crypto";
import { expect, test, type APIRequestContext } from "@playwright/test";
import { ADMIN_STORAGE_PATH } from "../../constants";
import { Page } from "../../fixtures/pages";
import { navigateToPage } from "../../helpers/navigation";
import { readBack } from "../../helpers/roundTrip";
import { masterKey } from "../../helpers/traffic";

const post = async (api: APIRequestContext, path: string, data: object) => {
  const response = await api.post(path, {
    data,
    headers: { Authorization: `Bearer ${masterKey()}` },
  });
  expect(response.ok(), `${path} returned ${response.status()}`).toBe(true);
  return response;
};

test.use({ storageState: ADMIN_STORAGE_PATH });

test("A user created in the key dialog becomes the owner of the new key", async ({ page, request }) => {
  const userId = `e2e-inline-${randomUUID()}`;
  let created = false;
  let keyCreated = false;
  try {
    await navigateToPage(page, Page.ApiKeys);
    await page.getByRole("button", { name: /Create New Key/ }).click();
    const keyDialog = page.getByRole("dialog", { name: "Create New Key" });
    await keyDialog.getByText("Another User", { exact: true }).click();
    await keyDialog.getByRole("button", { name: "Create User", exact: true }).click();
    const userDialog = page.getByRole("dialog", { name: "Create New User" });
    await userDialog.getByText("Supply custom User ID", { exact: true }).click();
    await userDialog.getByLabel("User ID", { exact: true }).fill(userId);
    const createdUser = page.waitForResponse(
      (response) => new URL(response.url()).pathname === "/user/new" && response.request().method() === "POST",
    );
    await userDialog.getByRole("button", { name: "Create User", exact: true }).click();
    expect((await createdUser).ok()).toBe(true);
    created = true;
    await expect(userDialog).toBeHidden();
    await expect(keyDialog.getByPlaceholder("Type email or user ID to search for users")).toHaveValue(userId);

    await keyDialog.getByLabel(/Key Name/).fill(userId);
    const createdKey = page.waitForResponse(
      (response) => new URL(response.url()).pathname === "/key/generate" && response.request().method() === "POST",
    );
    await keyDialog.getByRole("button", { name: "Create Key", exact: true }).click();
    const response = await createdKey;
    expect(response.ok()).toBe(true);
    keyCreated = true;
    await expect(page.getByRole("dialog", { name: "Save your Key" })).toBeVisible();
    const saved = await readBack<{ keys: { user_id: string; key_alias: string }[] }>(
      page,
      `/key/list?key_alias=${encodeURIComponent(userId)}&return_full_object=true&size=100`,
    );
    expect(saved.keys).toContainEqual(expect.objectContaining({ user_id: userId, key_alias: userId }));
  } finally {
    if (keyCreated) await post(request, "/key/delete", { key_aliases: [userId] });
    if (created) await post(request, "/user/delete", { user_ids: [userId] });
  }
});

test("Bulk edit updates the selected user's budget and adds them to both chosen teams", async ({ page, request }) => {
  const userId = `e2e-bulk-${randomUUID()}`;
  const teams: { id: string; alias: string }[] = [];
  let created = false;
  try {
    await post(request, "/user/new", {
      user_id: userId,
      user_role: "internal_user",
      auto_create_key: false,
      send_invite_email: false,
    });
    created = true;
    for (const suffix of ["first", "second"]) {
      const alias = `${userId}-${suffix}`;
      const response = await post(request, "/team/new", { team_alias: alias, team_id: alias });
      teams.push({ alias, id: (await response.json()).team_id });
    }
    await navigateToPage(page, Page.Users);
    await page.getByTestId("datatable-search").fill(userId);
    await page.getByTestId("toggle-user-selection").click();
    await page.getByRole("row").filter({ hasText: userId }).getByRole("checkbox").check();
    await page.getByRole("button", { name: /Bulk Edit/ }).click();
    const dialog = page.getByRole("dialog", { name: /Bulk Edit/ });
    await dialog.getByText("Unlimited Budget", { exact: true }).click();
    await dialog.getByPlaceholder("Enter a numerical value").fill("25");
    await dialog.getByText("Add selected users to teams", { exact: true }).click();
    const teamPicker = dialog.getByRole("combobox", { name: "Select teams to add users to" });
    await teamPicker.click();
    const options = teams.map((team) => page.getByRole("option", { name: team.alias, exact: true }));
    await Promise.all(options.map((option) => expect(option).toBeVisible()));
    for (const option of options) await option.click();
    await page.keyboard.press("Escape");

    await dialog.getByRole("button", { name: "Save Changes", exact: true }).click();
    await expect(dialog).toBeHidden();
    await expect(page.getByText(/Updated 1 user.*Added users to 2 team/)).toBeVisible();
    const user = await readBack<{ user_info: { max_budget: number } }>(
      page,
      `/user/info?user_id=${encodeURIComponent(userId)}`,
    );
    expect(user.user_info.max_budget).toBe(25);
    for (const { id } of teams) {
      const team = await readBack<{ team_info: { members_with_roles: { user_id: string; role: string }[] } }>(
        page,
        `/team/info?team_id=${encodeURIComponent(id)}`,
      );
      expect(team.team_info.members_with_roles).toContainEqual(expect.objectContaining({ user_id: userId, role: "user" }));
    }
  } finally {
    if (teams.length > 0) {
      await post(request, "/team/delete", { team_ids: teams.map(({ id }) => id) });
    }
    if (created) await post(request, "/user/delete", { user_ids: [userId] });
  }
});
