import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Organization, organizationInfoCall, organizationListCall } from "@/components/networking";
import { useQuery, useQueryClient, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";

const organizationKeys = createQueryKeys("organizations");
export const useOrganizations = (): UseQueryResult<Organization[]> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<Organization[]>({
    queryKey: organizationKeys.list({}),
    queryFn: async () => await organizationListCall(accessToken!),
    enabled: Boolean(accessToken && userId && userRole),
  });
};

export const useOrganization = (organizationID?: string) => {
  const queryClient = useQueryClient();
  const { accessToken } = useAuthorized();
  return useQuery<Organization>({
    queryKey: organizationKeys.detail(organizationID!),
    enabled: Boolean(accessToken && organizationID),

    queryFn: async () => {
      if (!accessToken || !organizationID) {
        throw new Error("Missing auth or teamId");
      }

      return organizationInfoCall(accessToken, organizationID);
    },

    initialData: () => {
      if (!organizationID) return undefined;

      const organizations = queryClient.getQueryData<Organization[]>(organizationKeys.list({}));

      return organizations?.find((organization: Organization) => organization.organization_id === organizationID);
    },
  });
};
