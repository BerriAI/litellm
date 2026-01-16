import { useQuery, useQueryClient, UseQueryResult } from "@tanstack/react-query";
import { Team } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchTeams } from "@/app/(dashboard)/networking";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";
import { teamInfoCall } from "@/components/networking";

const teamKeys = createQueryKeys("teams");
export const useTeams = (): UseQueryResult<Team[]> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<Team[]>({
    queryKey: teamKeys.list({}),
    queryFn: async () => await fetchTeams(accessToken!, userId, userRole, null),
    enabled: Boolean(accessToken),
  });
};

export const useTeam = (teamId?: string) => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  return useQuery<Team>({
    queryKey: teamKeys.detail(teamId!),
    enabled: Boolean(accessToken && teamId),

    queryFn: async () => {
      if (!accessToken || !teamId) {
        throw new Error("Missing auth or teamId");
      }

      return teamInfoCall(accessToken, teamId);
    },

    initialData: () => {
      if (!teamId) return undefined;

      const teams = queryClient.getQueryData<Team[]>(teamKeys.list({}));

      return teams?.find((team) => team.team_id === teamId);
    },
  });
};
