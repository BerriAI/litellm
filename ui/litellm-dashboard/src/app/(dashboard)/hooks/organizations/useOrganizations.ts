import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { organizationListCall, Organization } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const organizationKeys = createQueryKeys("organizations");

export const useOrganizations = (): UseQueryResult<Organization[]> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<Organization[]>({
    queryKey: organizationKeys.list({}),
    queryFn: async () => await organizationListCall(accessToken!),
    enabled: Boolean(accessToken && userId && userRole),
  });
};
