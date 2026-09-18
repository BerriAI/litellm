import { useMutation } from "@tanstack/react-query";
import { fetchClient } from "@/lib/http/api";

export interface ResetTeamMemberSpendParams {
  teamId: string;
  userId: string;
}

export const resetTeamMemberSpend = async ({ teamId, userId }: ResetTeamMemberSpendParams): Promise<void> => {
  await fetchClient.POST("/team/{team_id}/member/{user_id}/reset_spend", {
    params: { path: { team_id: teamId, user_id: userId } },
    body: { reset_to: 0 },
  });
};

export const useResetTeamMemberSpend = () =>
  useMutation<void, Error, ResetTeamMemberSpendParams>({ mutationFn: resetTeamMemberSpend });
