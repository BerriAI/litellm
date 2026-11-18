import { userConnectionsListCall, UserProviderConnectionsResponse } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export const userConnectionsKeys = createQueryKeys("userConnections");

export const useUserConnections = () => {
  const { accessToken } = useAuthorized();
  return useQuery<UserProviderConnectionsResponse>({
    queryKey: userConnectionsKeys.list({}),
    queryFn: async () => await userConnectionsListCall(accessToken!),
    enabled: Boolean(accessToken),
  });
};
