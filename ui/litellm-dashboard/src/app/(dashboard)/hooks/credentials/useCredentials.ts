import { credentialListCall, CredentialsResponse } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import useCan from "@/app/(dashboard)/hooks/useCan";

const credentialsKeys = createQueryKeys("credentials");

export const useCredentials = () => {
  const { accessToken } = useAuthorized();
  const canViewCredentials = useCan("viewCredentials");
  return useQuery<CredentialsResponse>({
    queryKey: credentialsKeys.list({}),
    queryFn: async () => await credentialListCall(accessToken!),
    enabled: Boolean(accessToken) && canViewCredentials,
  });
};
