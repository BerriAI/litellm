import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { AccessGroupResponse, accessGroupKeys } from "./useAccessGroups";

// ── Types ────────────────────────────────────────────────────────────────────

export interface AccessGroupUpdateParams {
  access_group_name?: string;
  description?: string | null;
  access_model_names?: string[];
  access_mcp_server_ids?: string[];
  access_agent_ids?: string[];
  assigned_team_ids?: string[];
  assigned_key_ids?: string[];
}

export interface EditAccessGroupVariables {
  accessGroupId: string;
  params: AccessGroupUpdateParams;
}

// ── Fetch function ───────────────────────────────────────────────────────────

const updateAccessGroup = async (
  accessToken: string,
  accessGroupId: string,
  params: AccessGroupUpdateParams,
): Promise<AccessGroupResponse> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/v1/access_group/${encodeURIComponent(accessGroupId)}`;

  const response = await fetch(url, {
    method: "PUT",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(params),
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

export const useEditAccessGroup = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<AccessGroupResponse, Error, EditAccessGroupVariables>({
    mutationFn: async ({ accessGroupId, params }) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return updateAccessGroup(accessToken, accessGroupId, params);
    },
    onSuccess: (_data, { accessGroupId }) => {
      queryClient.invalidateQueries({ queryKey: accessGroupKeys.all });
      queryClient.invalidateQueries({
        queryKey: accessGroupKeys.detail(accessGroupId),
      });
    },
  });
};
