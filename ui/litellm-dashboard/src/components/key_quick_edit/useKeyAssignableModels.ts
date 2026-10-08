import { useQuery } from "@tanstack/react-query";
import type { KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import { excludeProxyWideSentinel } from "@/components/key_team_helpers/fetch_available_models_team_key";
import { modelAvailableCall } from "@/components/networking";
import { fetchTeamModels } from "@/components/organisms/create_key_button";

type KeyModelScope = Pick<KeyResponse, "team_id">;
type TeamModelScope = Pick<Team, "team_id" | "models">;

export const useKeyAssignableModels = ({
  keyData,
  team,
  userId,
  userRole,
  accessToken,
  enabled,
}: {
  keyData: KeyModelScope;
  team: TeamModelScope | null | undefined;
  userId: string | null;
  userRole: string | null;
  accessToken: string | null;
  enabled: boolean;
}) =>
  useQuery<string[]>({
    queryKey: ["keyAssignableModels", keyData.team_id, team?.team_id, team?.models, userId, userRole],
    enabled: enabled && Boolean(accessToken && userId && userRole),
    queryFn: async () => {
      if (!accessToken || !userId || !userRole) return [];

      if (keyData.team_id == null) {
        const response = (await modelAvailableCall(accessToken, userId, userRole)) as { data: { id: string }[] };
        return excludeProxyWideSentinel(response.data.map((model) => model.id));
      }

      if (!team?.team_id) return [];

      const availableModels = await fetchTeamModels(userId, userRole, accessToken, team.team_id);
      return excludeProxyWideSentinel(Array.from(new Set([...team.models, ...availableModels])));
    },
  });
