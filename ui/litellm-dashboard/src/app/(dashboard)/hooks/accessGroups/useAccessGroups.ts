import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { getProxyBaseUrl, getGlobalLitellmHeaderName, deriveErrorMessage, handleError } from "@/components/networking";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import type { components } from "@/lib/http/schema";

// ── Types ────────────────────────────────────────────────────────────────────

export type AccessGroupResponse = components["schemas"]["AccessGroupResponse"];

// ── Query keys (shared across access-group hooks) ────────────────────────────

export const accessGroupKeys = createQueryKeys("accessGroups");

// ── Fetch function ───────────────────────────────────────────────────────────

const fetchAccessGroups = async (accessToken: string): Promise<AccessGroupResponse[]> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/v1/access_group`;

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

export const useAccessGroups = () => {
  const { accessToken, userRole } = useAuthorized();

  return useQuery<AccessGroupResponse[]>({
    queryKey: accessGroupKeys.list({}),
    queryFn: async () => fetchAccessGroups(accessToken!),
    enabled: Boolean(accessToken) && all_admin_roles.includes(userRole || ""),
  });
};
