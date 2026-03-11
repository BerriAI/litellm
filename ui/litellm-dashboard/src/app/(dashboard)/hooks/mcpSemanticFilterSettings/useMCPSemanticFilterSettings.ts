import { getMCPSemanticFilterSettings } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import useAuthorized from "../useAuthorized";

const mcpSemanticFilterSettingsKeys = createQueryKeys(
  "mcpSemanticFilterSettings"
);

export const useMCPSemanticFilterSettings = () => {
  const { accessToken } = useAuthorized();
  return useQuery<Record<string, any>>({
    queryKey: mcpSemanticFilterSettingsKeys.list({}),
    queryFn: async () => await getMCPSemanticFilterSettings(accessToken),
    enabled: !!accessToken,
    staleTime: 60 * 60 * 1000, // 1 hour
    gcTime: 60 * 60 * 1000, // 1 hour
  });
};
