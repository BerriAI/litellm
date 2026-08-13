import * as path from "path";

export const UI_BASE_URL = (
  process.env.E2E_UI_BASE_URL ||
  process.env.LITELLM_PROXY_URL ||
  "http://localhost:4000"
).replace(/\/+$/, "");

// Directory every artifact this suite writes must land in: storage states,
// failure screenshots, playwright output. A local `./run_e2e.sh` run has a
// writable cwd, so it keeps the historical behavior of writing beside the
// suite. In the packaged e2e image the suite ships on a read-only filesystem
// (/app/e2e/ui), so bare relative paths raise EROFS/ENOENT and the run dies in
// globalSetup before a single test executes. Point E2E_UI_ARTIFACT_DIR at a
// writable path (the image runner already exports TMPDIR) to relocate them.
export const ARTIFACT_DIR = process.env.E2E_UI_ARTIFACT_DIR || ".";

const storagePath = (name: string): string => path.join(ARTIFACT_DIR, name);

// Storage state paths for each role
export const ADMIN_STORAGE_PATH = storagePath("admin.storageState.json");
export const ADMIN_VIEWER_STORAGE_PATH = storagePath("adminViewer.storageState.json");
export const INTERNAL_USER_STORAGE_PATH = storagePath("internalUser.storageState.json");
export const INTERNAL_VIEWER_STORAGE_PATH = storagePath("internalViewer.storageState.json");
export const TEAM_ADMIN_STORAGE_PATH = storagePath("teamAdmin.storageState.json");

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
