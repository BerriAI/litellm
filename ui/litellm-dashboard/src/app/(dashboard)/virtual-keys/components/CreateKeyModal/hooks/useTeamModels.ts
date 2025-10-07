import { useEffect, useMemo, useState } from "react";
import type { Team } from "@/components/key_team_helpers/key_list";
import { fetchTeamModels } from "@/app/(dashboard)/virtual-keys/components/CreateKeyModal/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export const useTeamModels = (selectedTeam: Team | null) => {
  const { userId: userID, userRole, accessToken } = useAuthorized();
  const [modelsToPick, setModelsToPick] = useState<string[]>([]);

  // turn array into a stable dep so it only re-runs when contents change
  const selectedTeamModelsKey = useMemo(() => (selectedTeam?.models ?? []).join("|"), [selectedTeam?.models]);

  useEffect(() => {
    if (!userID || !userRole || !accessToken) {
      setModelsToPick([]);
      return;
    }
    (async () => {
      const fetched = await fetchTeamModels(userID, userRole, accessToken, selectedTeam?.team_id ?? null);
      const union = Array.from(new Set([...(selectedTeam?.models ?? []), ...(fetched || [])]));
      setModelsToPick(union);
    })();
  }, [userID, userRole, accessToken, selectedTeam?.team_id, selectedTeamModelsKey, selectedTeam?.models]);

  return modelsToPick;
};
