import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { projectKeys } from "./useProjects";

// ── Fetch function ───────────────────────────────────────────────────────────

const deleteProjects = async (
  accessToken: string,
  projectIds: string[],
): Promise<void> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/project/delete`;

  const response = await fetch(url, {
    method: "DELETE",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ project_ids: projectIds }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    const errorMessage = deriveErrorMessage(errorData);
    handleError(errorMessage);
    throw new Error(errorMessage);
  }
};

// ── Hook ─────────────────────────────────────────────────────────────────────

export const useDeleteProject = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<void, Error, string[]>({
    mutationFn: async (projectIds) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return deleteProjects(accessToken, projectIds);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.all });
    },
  });
};
