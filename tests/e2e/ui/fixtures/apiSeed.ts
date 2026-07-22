import { APIRequestContext, request } from "@playwright/test";
import { E2E_UI_OPENAI_MODEL, E2E_UI_ANTHROPIC_MODEL } from "../constants";

const OPENAI_UPSTREAM = `openai/${process.env.E2E_CHEAP_OPENAI_MODEL ?? "gpt-5.5"}`;
const ANTHROPIC_UPSTREAM = `anthropic/${process.env.E2E_CHEAP_ANTHROPIC_MODEL ?? "claude-haiku-4-5"}`;

const SEED_PASSWORD = "test";

const MODELS = [
  {
    model_name: E2E_UI_OPENAI_MODEL,
    litellm_params: { model: OPENAI_UPSTREAM, api_key: "os.environ/OPENAI_API_KEY" },
    model_info: { id: E2E_UI_OPENAI_MODEL },
  },
  {
    model_name: E2E_UI_ANTHROPIC_MODEL,
    litellm_params: { model: ANTHROPIC_UPSTREAM, api_key: "os.environ/ANTHROPIC_API_KEY" },
    model_info: { id: E2E_UI_ANTHROPIC_MODEL },
  },
] as const;

const ORG = { organization_id: "e2e-org-main", organization_alias: "E2E Organization", max_budget: 1000 } as const;

const USERS = [
  { user_id: "e2e-proxy-admin", user_email: "admin@test.local", user_role: "proxy_admin" },
  { user_id: "e2e-admin-viewer", user_email: "adminviewer@test.local", user_role: "proxy_admin_viewer" },
  { user_id: "e2e-internal-user", user_email: "internal@test.local", user_role: "internal_user" },
  { user_id: "e2e-internal-viewer", user_email: "viewer@test.local", user_role: "internal_user_viewer" },
  { user_id: "e2e-team-admin", user_email: "teamadmin@test.local", user_role: "internal_user" },
  { user_id: "e2e-invitable-user", user_email: "invitable@test.local", user_role: "internal_user" },
  { user_id: "e2e-internal-noteam", user_email: "noteam@test.local", user_role: "internal_user" },
  { user_id: "e2e-invitable-by-team-admin", user_email: "invitable-team@test.local", user_role: "internal_user" },
  { user_id: "e2e-removable-member", user_email: "removable@test.local", user_role: "internal_user" },
] as const;

const TEAMS = [
  {
    team_id: "e2e-team-crud",
    team_alias: "E2E Team CRUD",
    organization_id: null,
    models: [E2E_UI_OPENAI_MODEL, E2E_UI_ANTHROPIC_MODEL],
    members_with_roles: [
      { role: "admin", user_id: "e2e-team-admin" },
      { role: "user", user_id: "e2e-internal-user" },
      { role: "user", user_id: "e2e-internal-viewer" },
      { role: "user", user_id: "e2e-removable-member" },
    ],
  },
  {
    team_id: "e2e-team-delete",
    team_alias: "E2E Team Delete",
    organization_id: null,
    models: [E2E_UI_OPENAI_MODEL],
    members_with_roles: [{ role: "admin", user_id: "e2e-team-admin" }],
  },
  {
    team_id: "e2e-team-org",
    team_alias: "E2E Team In Org",
    organization_id: ORG.organization_id,
    models: [E2E_UI_OPENAI_MODEL],
    members_with_roles: [{ role: "user", user_id: "e2e-internal-user" }],
  },
  {
    team_id: "e2e-team-no-admin",
    team_alias: "E2E Team No Admin",
    organization_id: null,
    models: [E2E_UI_OPENAI_MODEL],
    members_with_roles: [{ role: "user", user_id: "e2e-invitable-user" }],
  },
] as const;

const KEYS = [
  { key_alias: "e2eUpdateLimitsKey", user_id: "e2e-proxy-admin", team_id: "e2e-team-crud" },
  { key_alias: "e2eDeleteKey", user_id: "e2e-proxy-admin", team_id: "e2e-team-crud" },
  { key_alias: "e2eRegenerateKey", user_id: "e2e-proxy-admin", team_id: "e2e-team-crud" },
  { key_alias: "e2eInternalUserKey", user_id: "e2e-internal-user", team_id: "e2e-team-crud" },
  { key_alias: "e2eViewerKey", user_id: "e2e-internal-viewer", team_id: null },
] as const;

async function post(api: APIRequestContext, path: string, data: unknown, allowFailure = false): Promise<void> {
  const res = await api.post(path, { data });
  if (res.ok() || allowFailure) return;
  throw new Error(`Seeding ${path} failed (${res.status()}): ${await res.text()}`);
}

async function teardown(api: APIRequestContext): Promise<void> {
  for (const key of KEYS) {
    await post(api, "/key/delete", { key_aliases: [key.key_alias] }, true);
  }
  for (const team of TEAMS) {
    await post(api, "/team/delete", { team_ids: [team.team_id] }, true);
  }
  for (const user of USERS) {
    await post(api, "/user/delete", { user_ids: [user.user_id] }, true);
  }
  await api.delete("/organization/delete", { data: { organization_ids: [ORG.organization_id] } });
  for (const model of MODELS) {
    await post(api, "/model/delete", { id: model.model_info.id }, true);
  }
}

async function create(api: APIRequestContext): Promise<void> {
  for (const model of MODELS) {
    await post(api, "/model/new", model);
  }
  await post(api, "/organization/new", ORG);
  for (const user of USERS) {
    await post(api, "/user/new", { ...user, auto_create_key: false });
    await post(api, "/user/update", { user_id: user.user_id, password: SEED_PASSWORD });
  }
  for (const team of TEAMS) {
    await post(api, "/team/new", team);
  }
  for (const key of KEYS) {
    await post(api, "/key/generate", { ...key, models: [E2E_UI_OPENAI_MODEL] });
  }
}

export async function seedGateway(baseUrl: string, masterKey: string): Promise<void> {
  const api = await request.newContext({
    baseURL: baseUrl,
    extraHTTPHeaders: { Authorization: `Bearer ${masterKey}` },
  });
  try {
    await teardown(api);
    await create(api);
  } finally {
    await api.dispose();
  }
}
