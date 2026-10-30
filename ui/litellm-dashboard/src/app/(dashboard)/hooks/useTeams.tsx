import { useEffect, useState } from "react";
import { Team } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchAllTeamsPaged } from "@/app/(dashboard)/hooks/teams/useTeams";
import { teamListScopeUserId } from "@/utils/roles";

/**
 * @deprecated This hook is deprecated. Use the react-query implementation from `@/app/(dashboard)/hooks/teams/useTeams` instead.
 * This version will be removed in a future release.
 */
const useTeams = () => {
  const [teams, setTeams] = useState<Team[] | null>([]);
  const { accessToken, userId: userID, userRole } = useAuthorized();

  useEffect(() => {
    if (!accessToken) return;
    (async () => {
      const fetched = await fetchAllTeamsPaged(accessToken, { userID: teamListScopeUserId(userRole, userID) });
      setTeams(fetched);
    })();
  }, [accessToken, userID, userRole]);

  return { teams, setTeams };
};

export default useTeams;
