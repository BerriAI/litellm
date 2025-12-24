import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { organizationListCall, Organization } from "@/components/networking";

const organizationKeys = createQueryKeys("organizations");

export const useOrganizations = (accessToken: string | null): UseQueryResult<Organization[]> => {
  return useQuery<Organization[]>({
    queryKey: organizationKeys.list({}),
    queryFn: async () => await organizationListCall(accessToken!),
    enabled: Boolean(accessToken),
  });
};
