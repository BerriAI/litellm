import { useMutation } from "@tanstack/react-query";
import type { TeamMemberBudgetPatch } from "@/components/team/bulkMemberLimits";
import { fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

export type TeamMemberBudgetUpdateResult = components["schemas"]["TeamMemberBudgetUpdateResult"];

export interface BulkUpdateTeamMembersParams {
  teamId: string;
  members: TeamMemberBudgetPatch[];
}

export const bulkUpdateTeamMembers = async ({
  teamId,
  members,
}: BulkUpdateTeamMembersParams): Promise<TeamMemberBudgetUpdateResult[]> => {
  const { data } = await fetchClient.POST("/management/v1/teams/{team_id}/members/bulk_update", {
    params: { path: { team_id: teamId } },
    body: { members },
  });
  return data?.data ?? [];
};

export const useBulkUpdateTeamMembers = () =>
  useMutation<TeamMemberBudgetUpdateResult[], Error, BulkUpdateTeamMembersParams>({
    mutationFn: bulkUpdateTeamMembers,
  });
