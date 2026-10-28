import { modelCostMap } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

export const modelCostMapKeys = createQueryKeys("modelCostMap");

export const useModelCostMap = (enabled = true, catalogOnly = false) => {
  return useQuery<Record<string, any>>({
    enabled,
    queryKey: modelCostMapKeys.list(catalogOnly ? { filters: { catalog_only: "true" } } : {}),
    queryFn: async () => await modelCostMap(catalogOnly),
    staleTime: 60 * 1000, // 1 minute
    gcTime: 60 * 1000, // 1 minute
  });
};
