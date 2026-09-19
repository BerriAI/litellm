import { getWebSearchInterceptionSettings } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import useAuthorized from "../useAuthorized";

const webSearchInterceptionSettingsKeys = createQueryKeys("webSearchInterceptionSettings");

export const useWebSearchInterceptionSettings = () => {
  const { accessToken } = useAuthorized();
  return useQuery<Record<string, any>>({
    queryKey: webSearchInterceptionSettingsKeys.list({}),
    queryFn: async () => await getWebSearchInterceptionSettings(accessToken),
    enabled: !!accessToken,
    staleTime: 60 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  });
};
