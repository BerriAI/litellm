import type { APIRequestContext } from "@playwright/test";
import {
  E2E_ADMIN_VIEWER_EMAIL,
  E2E_ADMIN_VIEWER_USER_ID,
  E2E_DELETE_KEY_ALIAS,
  E2E_INTERNAL_NOTEAM_EMAIL,
  E2E_INTERNAL_NOTEAM_USER_ID,
  E2E_INTERNAL_USER_EMAIL,
  E2E_INTERNAL_USER_ID,
  E2E_INTERNAL_USER_KEY_ALIAS,
  E2E_INTERNAL_VIEWER_EMAIL,
  E2E_INTERNAL_VIEWER_USER_ID,
  E2E_INVITABLE_BY_TEAM_ADMIN_EMAIL,
  E2E_INVITABLE_BY_TEAM_ADMIN_USER_ID,
  E2E_INVITABLE_USER_EMAIL,
  E2E_INVITABLE_USER_ID,
  E2E_ORG_ALIAS,
  E2E_ORG_BUDGET_ID,
  E2E_ORG_ID,
  E2E_PROXY_ADMIN_EMAIL,
  E2E_PROXY_ADMIN_USER_ID,
  E2E_REGENERATE_KEY_ALIAS,
  E2E_REMOVABLE_MEMBER_EMAIL,
  E2E_REMOVABLE_MEMBER_USER_ID,
  E2E_TEAM_ADMIN_EMAIL,
  E2E_TEAM_ADMIN_USER_ID,
  E2E_TEAM_CRUD_ALIAS,
  E2E_TEAM_CRUD_ID,
  E2E_TEAM_DELETE_ALIAS,
  E2E_TEAM_DELETE_ID,
  E2E_TEAM_NO_ADMIN_ALIAS,
  E2E_TEAM_NO_ADMIN_ID,
  E2E_TEAM_ORG_ALIAS,
  E2E_TEAM_ORG_ID,
  E2E_UPDATE_LIMITS_KEY_ALIAS,
  E2E_USER_PASSWORD,
  E2E_VIEWER_KEY_ALIAS,
} from "../constants";

type UserRole = "proxy_admin" | "proxy_admin_viewer" | "internal_user" | "internal_user_viewer";
type TeamMemberRole = "admin" | "user";

type SeedUser = { readonly userId: string; readonly email: string; readonly role: UserRole };
type SeedTeamMember = { readonly user_id: string; readonly role: TeamMemberRole };
type SeedTeam = {
  readonly teamId: string;
  readonly alias: string;
  readonly organizationId: string | null;
  readonly models: readonly string[];
  readonly members: readonly SeedTeamMember[];
};
type SeedKey = {
  readonly alias: string;
  readonly userId: string;
  readonly teamId: string | null;
  readonly models: readonly string[];
};

const OPENAI_MODEL = "fake-openai-gpt-4";
const ANTHROPIC_MODEL = "fake-anthropic-claude";
const ORG_MAX_BUDGET = 1000;

// The master key's own user is auto-added as an admin of every team it creates,
// which would make the proxy admin a member of teams the suite needs it to be a
// stranger to. Removed after each team is created.
const MASTER_KEY_USER_ID = "default_user_id";

const USERS: readonly SeedUser[] = [
  { userId: E2E_PROXY_ADMIN_USER_ID, email: E2E_PROXY_ADMIN_EMAIL, role: "proxy_admin" },
  { userId: E2E_ADMIN_VIEWER_USER_ID, email: E2E_ADMIN_VIEWER_EMAIL, role: "proxy_admin_viewer" },
  { userId: E2E_INTERNAL_USER_ID, email: E2E_INTERNAL_USER_EMAIL, role: "internal_user" },
  { userId: E2E_INTERNAL_VIEWER_USER_ID, email: E2E_INTERNAL_VIEWER_EMAIL, role: "internal_user_viewer" },
  { userId: E2E_TEAM_ADMIN_USER_ID, email: E2E_TEAM_ADMIN_EMAIL, role: "internal_user" },
  { userId: E2E_INVITABLE_USER_ID, email: E2E_INVITABLE_USER_EMAIL, role: "internal_user" },
  { userId: E2E_INTERNAL_NOTEAM_USER_ID, email: E2E_INTERNAL_NOTEAM_EMAIL, role: "internal_user" },
  {
    userId: E2E_INVITABLE_BY_TEAM_ADMIN_USER_ID,
    email: E2E_INVITABLE_BY_TEAM_ADMIN_EMAIL,
    role: "internal_user",
  },
  { userId: E2E_REMOVABLE_MEMBER_USER_ID, email: E2E_REMOVABLE_MEMBER_EMAIL, role: "internal_user" },
];

const TEAMS: readonly SeedTeam[] = [
  {
    teamId: E2E_TEAM_CRUD_ID,
    alias: E2E_TEAM_CRUD_ALIAS,
    organizationId: null,
    models: [OPENAI_MODEL, ANTHROPIC_MODEL],
    members: [
      { user_id: E2E_TEAM_ADMIN_USER_ID, role: "admin" },
      { user_id: E2E_INTERNAL_USER_ID, role: "user" },
      { user_id: E2E_INTERNAL_VIEWER_USER_ID, role: "user" },
      { user_id: E2E_REMOVABLE_MEMBER_USER_ID, role: "user" },
    ],
  },
  {
    teamId: E2E_TEAM_DELETE_ID,
    alias: E2E_TEAM_DELETE_ALIAS,
    organizationId: null,
    models: [OPENAI_MODEL],
    members: [{ user_id: E2E_TEAM_ADMIN_USER_ID, role: "admin" }],
  },
  {
    teamId: E2E_TEAM_ORG_ID,
    alias: E2E_TEAM_ORG_ALIAS,
    organizationId: E2E_ORG_ID,
    models: [OPENAI_MODEL],
    members: [{ user_id: E2E_INTERNAL_USER_ID, role: "user" }],
  },
  {
    teamId: E2E_TEAM_NO_ADMIN_ID,
    alias: E2E_TEAM_NO_ADMIN_ALIAS,
    organizationId: null,
    models: [OPENAI_MODEL],
    members: [{ user_id: E2E_INVITABLE_USER_ID, role: "user" }],
  },
];

const KEYS: readonly SeedKey[] = [
  {
    alias: E2E_UPDATE_LIMITS_KEY_ALIAS,
    userId: E2E_PROXY_ADMIN_USER_ID,
    teamId: E2E_TEAM_CRUD_ID,
    models: [OPENAI_MODEL],
  },
  {
    alias: E2E_DELETE_KEY_ALIAS,
    userId: E2E_PROXY_ADMIN_USER_ID,
    teamId: E2E_TEAM_CRUD_ID,
    models: [OPENAI_MODEL],
  },
  {
    alias: E2E_REGENERATE_KEY_ALIAS,
    userId: E2E_PROXY_ADMIN_USER_ID,
    teamId: E2E_TEAM_CRUD_ID,
    models: [OPENAI_MODEL],
  },
  {
    alias: E2E_INTERNAL_USER_KEY_ALIAS,
    userId: E2E_INTERNAL_USER_ID,
    teamId: E2E_TEAM_CRUD_ID,
    models: [OPENAI_MODEL],
  },
  {
    alias: E2E_VIEWER_KEY_ALIAS,
    userId: E2E_INTERNAL_VIEWER_USER_ID,
    teamId: null,
    models: [OPENAI_MODEL],
  },
];

type JsonBody = Record<string, unknown>;

const jsonHeaders = (masterKey: string): Record<string, string> => ({
  Authorization: `Bearer ${masterKey}`,
  "Content-Type": "application/json",
});

async function post(
  api: APIRequestContext,
  url: string,
  data: JsonBody,
  headers: Record<string, string>,
): Promise<void> {
  const res = await api.post(url, { headers, data });
  if (!res.ok()) {
    throw new Error(`Seeding call POST ${url} failed (${res.status()}): ${await res.text()}`);
  }
}

async function postAllowingMissing(
  api: APIRequestContext,
  url: string,
  data: JsonBody,
  headers: Record<string, string>,
): Promise<void> {
  const res = await api.post(url, { headers, data });
  if (!res.ok() && res.status() !== 404) {
    throw new Error(`Seeding call POST ${url} failed (${res.status()}): ${await res.text()}`);
  }
}

async function deleteAllowingMissing(
  api: APIRequestContext,
  url: string,
  data: JsonBody,
  headers: Record<string, string>,
): Promise<void> {
  const res = await api.delete(url, { headers, data });
  if (!res.ok() && res.status() !== 404) {
    throw new Error(`Seeding call DELETE ${url} failed (${res.status()}): ${await res.text()}`);
  }
}

// /team/delete and /user/delete are all-or-nothing: one unknown id in the batch
// aborts the whole call and deletes nothing, so each id goes in its own request.
async function removeFixtures(
  api: APIRequestContext,
  apiBase: string,
  headers: Record<string, string>,
): Promise<void> {
  await postAllowingMissing(api, `${apiBase}/key/delete`, { key_aliases: KEYS.map((key) => key.alias) }, headers);
  for (const team of TEAMS) {
    await postAllowingMissing(api, `${apiBase}/team/delete`, { team_ids: [team.teamId] }, headers);
  }
  for (const user of USERS) {
    await postAllowingMissing(api, `${apiBase}/user/delete`, { user_ids: [user.userId] }, headers);
  }
  await deleteAllowingMissing(api, `${apiBase}/organization/delete`, { organization_ids: [E2E_ORG_ID] }, headers);
  await postAllowingMissing(api, `${apiBase}/budget/delete`, { id: E2E_ORG_BUDGET_ID }, headers);
}

async function createFixtures(
  api: APIRequestContext,
  apiBase: string,
  headers: Record<string, string>,
): Promise<void> {
  await post(api, `${apiBase}/budget/new`, { budget_id: E2E_ORG_BUDGET_ID, max_budget: ORG_MAX_BUDGET }, headers);
  await post(
    api,
    `${apiBase}/organization/new`,
    { organization_id: E2E_ORG_ID, organization_alias: E2E_ORG_ALIAS, budget_id: E2E_ORG_BUDGET_ID },
    headers,
  );

  for (const user of USERS) {
    await post(
      api,
      `${apiBase}/user/new`,
      { user_id: user.userId, user_email: user.email, user_role: user.role, auto_create_key: false },
      headers,
    );
    await post(api, `${apiBase}/user/update`, { user_id: user.userId, password: E2E_USER_PASSWORD }, headers);
  }

  for (const team of TEAMS) {
    await post(
      api,
      `${apiBase}/team/new`,
      {
        team_id: team.teamId,
        team_alias: team.alias,
        organization_id: team.organizationId,
        models: team.models,
        members_with_roles: team.members,
      },
      headers,
    );
    await postAllowingMissing(
      api,
      `${apiBase}/team/member_delete`,
      { team_id: team.teamId, user_id: MASTER_KEY_USER_ID },
      headers,
    );
  }

  for (const key of KEYS) {
    await post(
      api,
      `${apiBase}/key/generate`,
      { key_alias: key.alias, user_id: key.userId, team_id: key.teamId, models: key.models },
      headers,
    );
  }
}

export async function seedFixtures(api: APIRequestContext, apiBase: string, masterKey: string): Promise<void> {
  const headers = jsonHeaders(masterKey);
  await removeFixtures(api, apiBase, headers);
  await createFixtures(api, apiBase, headers);
}
