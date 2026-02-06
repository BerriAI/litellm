import { getProviderCreateMetadata, ProviderCreateInfo } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const providerFieldsKeys = createQueryKeys("providerFields");

export const useProviderFields = () => {
  return useQuery<ProviderCreateInfo[]>({
    queryKey: providerFieldsKeys.list({}),
    queryFn: async () => await getProviderCreateMetadata(),
    staleTime: 24 * 60 * 60 * 1000, // 24 hours - data rarely changes
    gcTime: 24 * 60 * 60 * 1000, // 24 hours - keep in cache for 24 hours
  });
};
