import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { Team } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchTeams } from "@/app/(dashboard)/networking";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";

const teamKeys = createQueryKeys("teams");

export const useTeams = (): UseQueryResult<Team[]> => {
  const { accessToken, userId, userRole } = useAuthorized();

  return useQuery<Team[]>({
    queryKey: teamKeys.list({}),
    queryFn: async () => await fetchTeams(accessToken!, userId, userRole, null),
    enabled: Boolean(accessToken),
  });
};
