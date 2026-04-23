import { modelCostMap } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const modelCostMapKeys = createQueryKeys("modelCostMap");

export const useModelCostMap = () => {
  return useQuery<Record<string, any>>({
    queryKey: modelCostMapKeys.list({}),
    queryFn: async () => await modelCostMap(),
    staleTime: 60 * 1000, // 1 minute
    gcTime: 60 * 1000, // 1 minute
  });
};
