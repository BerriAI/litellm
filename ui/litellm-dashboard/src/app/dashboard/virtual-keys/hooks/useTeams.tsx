import { useEffect, useState } from "react";
import { fetchTeams } from "@/app/dashboard/virtual-keys/networking";
import { Team } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/dashboard/hooks/useAuthorized";

const useTeams = () => {
  const [teams, setTeams] = useState<Team[]>([]);
  const { accessToken, userId: userID, userRole } = useAuthorized();

  useEffect(() => {
    (async () => {
      const fetched = await fetchTeams(accessToken, userID, userRole, null);
      setTeams(fetched);
    })();
  }, [accessToken, userID, userRole]);

  return teams;
};

export default useTeams;
