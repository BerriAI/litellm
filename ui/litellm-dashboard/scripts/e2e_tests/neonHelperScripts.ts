import { createApiClient, EndpointType } from "@neondatabase/api-client";
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
      endpoints: [
        {
          type: EndpointType.ReadWrite,
          autoscaling_limit_min_cu: 0.25,
          autoscaling_limit_max_cu: 1,
        },
      ],
    });
    return response;
  } catch (error) {
    throw error;
  }
}

export async function getNeonE2ETestingBranchConnectionString() {
  const createBranchResponse = await createNeonE2ETestingBranch(PROJECT_ID, PARENT_BRANCH);
  const projectId = createBranchResponse.data.branch.project_id;
  const response = await apiClient.getConnectionUri({
    database_name: NEON_E2E_UI_TEST_DB_NAME,
    role_name: "neondb_owner",
    projectId: projectId,
  });
  console.log("connection string:", response.data.uri);
  return response.data.uri;
}

getNeonE2ETestingBranchConnectionString();
