import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { createApiClient } from "@/lib/http/client";

export interface TeamMemberBulkUpdateFields {
  role?: "admin" | "user" | null;
  max_budget_in_team?: number | null;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
  budget_duration?: string | null;
  allowed_models?: string[] | null;
}

const apiClient = createApiClient({
  getBaseUrl: getProxyBaseUrl,
  getAuthHeaderName: getGlobalLitellmHeaderName,
});

export const teamMemberBulkUpdateCall = async (
  accessToken: string,
  teamId: string,
  userIds: string[],
  updateFields: TeamMemberBulkUpdateFields,
) =>
  apiClient.patch(`/v2/team/${encodeURIComponent(teamId)}/members`, {
    accessToken,
    body: {
      user_ids: userIds,
      update_fields: updateFields,
    },
  });
