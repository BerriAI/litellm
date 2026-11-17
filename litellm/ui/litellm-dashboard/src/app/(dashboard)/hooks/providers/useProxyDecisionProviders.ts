import { getGuardrailUISettings } from "@/components/networking";
import { parseProxyDecisionProviders } from "@/lib/decisionModels";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const proxyDecisionProvidersKeys = createQueryKeys("proxyDecisionProviders");

export const useProxyDecisionProviders = (accessToken: string | null) => {
  const query = {
    queryKey: proxyDecisionProvidersKeys.list({}),
    queryFn: async () => parseProxyDecisionProviders(await getGuardrailUISettings(accessToken ?? "")),
    enabled: accessToken !== null,
    retry: false,
    staleTime: 24 * 60 * 60 * 1000,
  };
  return useQuery(query);
};
