import { credentialListCall, CredentialsResponse } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const credentialsKeys = createQueryKeys("credentials");

export const useCredentials = (accessToken: string | null) => {
  return useQuery<CredentialsResponse>({
    queryKey: credentialsKeys.list({}),
    queryFn: async () => await credentialListCall(accessToken!),
    enabled: Boolean(accessToken),
  });
};
