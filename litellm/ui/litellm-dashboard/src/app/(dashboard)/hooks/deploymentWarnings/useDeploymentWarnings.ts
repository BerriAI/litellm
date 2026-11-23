import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { backgroundFetchClient, fetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";
import { isProxyAdminRole, isProxyAdminTierRole } from "@/utils/roles";
import { createQueryKeys } from "../common/queryKeysFactory";

export type DeploymentWarning = components["schemas"]["DeploymentWarning"];
export type DeploymentWarningsResponse = components["schemas"]["DeploymentWarningsResponse"];
export type DeploymentWarningIgnoreUpdate = components["schemas"]["DeploymentWarningIgnoreUpdate"];

const deploymentWarningKeys = createQueryKeys("deploymentWarnings");

const fetchDeploymentWarnings = async (): Promise<DeploymentWarningsResponse> => {
  const { data } = await backgroundFetchClient.GET("/health/deployment_warnings");
  if (data === undefined) {
    throw new Error("The proxy returned no deployment warnings payload");
  }
  return data;
};

export const useDeploymentHealthAccess = () => {
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  return {
    canView: Boolean(accessToken) && isProxyAdminTierRole(userRole),
    canManage: Boolean(accessToken) && isProxyAdminRole(userRole) && !isViewOnly,
  };
};

export const useDeploymentWarnings = ({ refetchOnOpen = false }: { refetchOnOpen?: boolean } = {}) => {
  const { canView } = useDeploymentHealthAccess();
  const options = {
    queryKey: deploymentWarningKeys.all,
    queryFn: fetchDeploymentWarnings,
    enabled: canView,
    staleTime: 5 * 60 * 1000,
    retry: false,
    ...(refetchOnOpen ? { refetchOnMount: "always" as const } : {}),
  };
  return useQuery(options);
};

export const useUpdateDeploymentWarningIgnore = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ warningId, update }: { warningId: string; update: DeploymentWarningIgnoreUpdate }) => {
      const { data } = await fetchClient.PATCH("/health/deployment_warnings/{warning_id}", {
        params: { path: { warning_id: warningId } },
        body: update,
      });
      return data;
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: deploymentWarningKeys.all }),
    onError: (error) => toast.fromError(error),
  });
};
