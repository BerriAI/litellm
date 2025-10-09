import { useEffect, useState } from "react";
import { Team } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchTeams } from "@/app/(dashboard)/networking";

const useTeams = () => {
  const [teams, setTeams] = useState<Team[] | null>([]);
  const { accessToken, userId: userID, userRole } = useAuthorized();

  useEffect(() => {
    (async () => {
      const fetched = await fetchTeams(accessToken, userID, userRole, null);
      setTeams(fetched);
    })();
  }, [accessToken, userID, userRole]);

  return { teams, setTeams };
};

export default useTeams;
