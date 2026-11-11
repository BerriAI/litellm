import { useMutation } from "@tanstack/react-query";
import { fetchClient } from "@/lib/http/api";

export interface ResetTeamMemberBudgetParams {
  teamId: string;
  userId: string;
}

export const resetTeamMemberBudget = async ({ teamId, userId }: ResetTeamMemberBudgetParams): Promise<void> => {
  await fetchClient.POST("/team/{team_id}/member/{user_id}/reset_budget", {
    params: { path: { team_id: teamId, user_id: userId } },
  });
};

export const useResetTeamMemberBudget = () =>
  useMutation<void, Error, ResetTeamMemberBudgetParams>({ mutationFn: resetTeamMemberBudget });
