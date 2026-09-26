import { getCyberArkConfig, type CyberArkConfigResponse } from "./cyberArkApi";
import { useQuery } from "@tanstack/react-query";
import useAuthorized from "../useAuthorized";
import { createQueryKeys } from "../common/queryKeysFactory";

export const cyberArkKeys = createQueryKeys("cyberArkConfig");

export const useCyberArkConfig = () => {
  const { accessToken } = useAuthorized();

  const queryOptions = {
    queryKey: cyberArkKeys.list({}),
    queryFn: async () => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return getCyberArkConfig(accessToken);
    },
    enabled: !!accessToken,
    staleTime: 60 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  };
  return useQuery<CyberArkConfigResponse>(queryOptions);
};
