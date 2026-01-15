import { createApiClient } from "@neondatabase/api-client";
import { config } from "dotenv";
import { resolve } from "path";

const envPaths = [
  resolve(process.cwd(), "../../.env"), // project root
];

for (const envPath of envPaths) {
  config({ path: envPath });
}

const NEON_API_KEY = process.env.NEON_API_KEY!;
const PROJECT_ID = process.env.NEON_PROJECT_ID!;
const PARENT_BRANCH = process.env.NEON_PARENT_BRANCH_ID!;
const NEON_E2E_UI_TEST_DB_NAME = process.env.NEON_E2E_UI_TEST_DB_NAME!;

const apiClient = createApiClient({
  apiKey: NEON_API_KEY,
});

export async function createNeonE2ETestingBranch(projectId: string, parentBranchId?: string, expireAt?: string) {
  try {
    const response = await apiClient.createProjectBranch(projectId, {
      branch: {
        name: `e2e-local-${crypto.randomUUID()}`,
        parent_id: parentBranchId,
        expires_at: expireAt ?? new Date(Date.now() + 1000 * 60 * 30).toISOString(),
      },
    });
    return response;
  } catch (error) {
    throw error;
  }
}

export async function getNeonE2ETestingBranchConnectionString() {
  await createNeonE2ETestingBranch(PROJECT_ID, PARENT_BRANCH);

  const response = await apiClient.getConnectionUri({
    database_name: NEON_E2E_UI_TEST_DB_NAME,
    role_name: "neondb_owner",
    projectId: PROJECT_ID,
  });
  console.log("connection string:", response.data.uri);
  return response.data.uri;
}
