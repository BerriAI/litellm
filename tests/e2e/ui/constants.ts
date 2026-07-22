export const PROXY_BASE_URL = (process.env.LITELLM_PROXY_URL ?? "http://localhost:4000").replace(/\/+$/, "");

export const E2E_UI_OPENAI_MODEL = "e2e-ui-openai";
export const E2E_UI_ANTHROPIC_MODEL = "e2e-ui-anthropic";

// Storage state paths for each role
export const ADMIN_STORAGE_PATH = "admin.storageState.json";
export const ADMIN_VIEWER_STORAGE_PATH = "adminViewer.storageState.json";
export const INTERNAL_USER_STORAGE_PATH = "internalUser.storageState.json";
export const INTERNAL_VIEWER_STORAGE_PATH = "internalViewer.storageState.json";
export const TEAM_ADMIN_STORAGE_PATH = "teamAdmin.storageState.json";

// Seeded user identities (match fixtures/apiSeed.ts)
export const E2E_PROXY_ADMIN_USER_ID = "e2e-proxy-admin";
export const E2E_PROXY_ADMIN_EMAIL = "admin@test.local";
export const E2E_INTERNAL_USER_ID = "e2e-internal-user";
export const E2E_INTERNAL_USER_EMAIL = "internal@test.local";

// Key aliases for seeded test keys (match fixtures/apiSeed.ts)
export const E2E_UPDATE_LIMITS_KEY_ALIAS = "e2eUpdateLimitsKey";
export const E2E_DELETE_KEY_ALIAS = "e2eDeleteKey";
export const E2E_REGENERATE_KEY_ALIAS = "e2eRegenerateKey";
export const E2E_INTERNAL_USER_KEY_ALIAS = "e2eInternalUserKey";
export const E2E_VIEWER_KEY_ALIAS = "e2eViewerKey";

// Team identifiers (match fixtures/apiSeed.ts)
export const E2E_TEAM_CRUD_ID = "e2e-team-crud";
export const E2E_TEAM_CRUD_ALIAS = "E2E Team CRUD";
export const E2E_TEAM_DELETE_ID = "e2e-team-delete";
export const E2E_TEAM_DELETE_ALIAS = "E2E Team Delete";
export const E2E_TEAM_ORG_ID = "e2e-team-org";
export const E2E_TEAM_ORG_ALIAS = "E2E Team In Org";
export const E2E_TEAM_NO_ADMIN_ID = "e2e-team-no-admin";
export const E2E_TEAM_NO_ADMIN_ALIAS = "E2E Team No Admin";
