import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { AccessGroupResponse, accessGroupKeys } from "./useAccessGroups";

// ── Fetch function ───────────────────────────────────────────────────────────

const fetchAccessGroupDetails = async (
  accessToken: string,
  accessGroupId: string,
): Promise<AccessGroupResponse> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/v1/access_group/${encodeURIComponent(accessGroupId)}`;

  const response = await fetch(url, {
    method: "GET",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });

  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    handleError(errorMessage);
    throw new Error(errorMessage);
  }

  return response.json();
};

// ── Hook ─────────────────────────────────────────────────────────────────────

export const useAccessGroupDetails = (accessGroupId?: string) => {
  const { accessToken, userRole } = useAuthorized();
  const queryClient = useQueryClient();

  return useQuery<AccessGroupResponse>({
    queryKey: accessGroupKeys.detail(accessGroupId!),
    queryFn: async () => fetchAccessGroupDetails(accessToken!, accessGroupId!),
    enabled:
      Boolean(accessToken && accessGroupId) &&
      all_admin_roles.includes(userRole || ""),

    // Seed from the list cache when available
    initialData: () => {
      if (!accessGroupId) return undefined;

      const groups = queryClient.getQueryData<AccessGroupResponse[]>(
        accessGroupKeys.list({}),
      );

      return groups?.find((g) => g.access_group_id === accessGroupId);
    },
  });
};
