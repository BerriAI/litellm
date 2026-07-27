// Target base URL for the proxy and its admin UI. Derived from the environment
// so the suite can run against a deployed proxy as well as a local one; the
// previous hardcoded http://localhost:4000 made it local-only.
export const BASE_URL = (process.env.LITELLM_PROXY_URL ?? "http://localhost:4000").replace(/\/+$/, "");

// Where Playwright storage state is written. cwd is read-only in the e2e runner
// image (readOnlyRootFilesystem), so allow redirecting it there while keeping
// the repo-relative default for local runs.
export const STATE_DIR = (process.env.PLAYWRIGHT_STATE_DIR ?? ".").replace(/\/+$/, "");

// Storage state paths for each role
export const ADMIN_STORAGE_PATH = `${STATE_DIR}/admin.storageState.json`;
export const ADMIN_VIEWER_STORAGE_PATH = `${STATE_DIR}/adminViewer.storageState.json`;
export const INTERNAL_USER_STORAGE_PATH = `${STATE_DIR}/internalUser.storageState.json`;
export const INTERNAL_VIEWER_STORAGE_PATH = `${STATE_DIR}/internalViewer.storageState.json`;
export const TEAM_ADMIN_STORAGE_PATH = `${STATE_DIR}/teamAdmin.storageState.json`;

// Seeded user identities (match seed.sql)
export const E2E_PROXY_ADMIN_USER_ID = "e2e-proxy-admin";
export const E2E_PROXY_ADMIN_EMAIL = "admin@test.local";
export const E2E_INTERNAL_USER_ID = "e2e-internal-user";
export const E2E_INTERNAL_USER_EMAIL = "internal@test.local";

// Key aliases for seeded test keys (match seed.sql)
export const E2E_UPDATE_LIMITS_KEY_ALIAS = "e2eUpdateLimitsKey";
export const E2E_DELETE_KEY_ALIAS = "e2eDeleteKey";
export const E2E_REGENERATE_KEY_ALIAS = "e2eRegenerateKey";
export const E2E_INTERNAL_USER_KEY_ALIAS = "e2eInternalUserKey";
export const E2E_VIEWER_KEY_ALIAS = "e2eViewerKey";

// Team identifiers (match seed.sql)
export const E2E_TEAM_CRUD_ID = "e2e-team-crud";
export const E2E_TEAM_CRUD_ALIAS = "E2E Team CRUD";
export const E2E_TEAM_DELETE_ID = "e2e-team-delete";
export const E2E_TEAM_DELETE_ALIAS = "E2E Team Delete";
export const E2E_TEAM_ORG_ID = "e2e-team-org";
export const E2E_TEAM_ORG_ALIAS = "E2E Team In Org";
export const E2E_TEAM_NO_ADMIN_ID = "e2e-team-no-admin";
export const E2E_TEAM_NO_ADMIN_ALIAS = "E2E Team No Admin";
