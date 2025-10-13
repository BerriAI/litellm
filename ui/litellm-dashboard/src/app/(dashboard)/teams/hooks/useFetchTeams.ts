import { useCallback, useEffect, useState } from "react";
import { fetchTeams } from "@/components/common_components/fetch_teams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Organization, Team } from "@/components/networking";

interface useFetchTeamsProps {
  currentOrg: Organization | null;
  setTeams: (teams: Team[] | null) => void;
}

const useFetchTeams = ({ currentOrg, setTeams }: useFetchTeamsProps) => {
  const [lastRefreshed, setLastRefreshed] = useState("");
  const { accessToken, userId, userRole } = useAuthorized();

  const onRefreshClick = useCallback(() => {
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  }, []);

  useEffect(() => {
    if (accessToken) {
      fetchTeams(accessToken, userId, userRole, currentOrg, setTeams).then();
    }
    onRefreshClick();
  }, [accessToken, currentOrg, lastRefreshed, onRefreshClick, setTeams, userId, userRole]);

  return { lastRefreshed, setLastRefreshed, onRefreshClick };
};

export default useFetchTeams;
