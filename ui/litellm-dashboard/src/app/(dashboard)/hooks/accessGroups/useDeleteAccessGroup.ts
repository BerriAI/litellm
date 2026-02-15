import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { accessGroupKeys } from "./useAccessGroups";

// ── Fetch function ───────────────────────────────────────────────────────────

const deleteAccessGroup = async (
  accessToken: string,
  accessGroupId: string,
): Promise<void> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/v1/access_group/${encodeURIComponent(accessGroupId)}`;

  const response = await fetch(url, {
    method: "DELETE",
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

  // 204 No Content — nothing to parse
};

// ── Hook ─────────────────────────────────────────────────────────────────────

export const useDeleteAccessGroup = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<void, Error, string>({
    mutationFn: async (accessGroupId) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return deleteAccessGroup(accessToken, accessGroupId);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: accessGroupKeys.all });
    },
  });
};
