import { useQuery } from "@tanstack/react-query";

import { getAutoRouterRecommendation, type AutoRouterRecommendationResponse } from "@/components/networking";
import type { AutoSetupObjective, AutoSetupQualityLevel } from "@/components/add_model/build_complexity_router_config";
import { createQueryKeys } from "../common/queryKeysFactory";

const recommendationKeys = createQueryKeys("autoRouterRecommendation");

interface UseAutoRouterRecommendationParams {
  accessToken: string;
  qualityLevel: AutoSetupQualityLevel;
  optimizeFor: AutoSetupObjective;
  teamId?: string;
  enabled: boolean;
}

export const useAutoRouterRecommendation = ({
  accessToken,
  qualityLevel,
  optimizeFor,
  teamId,
  enabled,
}: UseAutoRouterRecommendationParams) => {
  const filters = { qualityLevel, optimizeFor, ...(teamId ? { teamId } : {}) };
  const queryOptions = {
    queryKey: recommendationKeys.list({
      filters,
    }),
    queryFn: () => getAutoRouterRecommendation(accessToken, qualityLevel, optimizeFor, teamId),
    enabled: enabled && Boolean(accessToken),
    staleTime: 30 * 1000,
  };
  return useQuery<AutoRouterRecommendationResponse>(queryOptions);
};
