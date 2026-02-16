import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

// ── Types ────────────────────────────────────────────────────────────────────

export interface AccessGroupResponse {
  access_group_id: string;
  access_group_name: string;
  description: string | null;
  access_model_names: string[];
  access_mcp_server_ids: string[];
  access_agent_ids: string[];
  assigned_team_ids: string[];
  assigned_key_ids: string[];
  created_at: string;
  created_by: string | null;
  updated_at: string;
  updated_by: string | null;
}

// ── Query keys (shared across access-group hooks) ────────────────────────────

export const accessGroupKeys = createQueryKeys("accessGroups");

// ── Fetch function ───────────────────────────────────────────────────────────

const fetchAccessGroups = async (
  accessToken: string,
): Promise<AccessGroupResponse[]> => {
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
    enabled:
      Boolean(accessToken) && all_admin_roles.includes(userRole || ""),
  });
};
