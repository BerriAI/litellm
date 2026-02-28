import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { ProjectResponse, projectKeys } from "./useProjects";

// ── Types ────────────────────────────────────────────────────────────────────

export interface ProjectUpdateParams {
  project_alias?: string;
  description?: string;
  team_id?: string;
  models?: string[];
  max_budget?: number;
  blocked?: boolean;
  metadata?: Record<string, unknown>;
  model_rpm_limit?: Record<string, number>;
  model_tpm_limit?: Record<string, number>;
}

// ── Fetch function ───────────────────────────────────────────────────────────

const updateProject = async (
  accessToken: string,
  projectId: string,
  params: ProjectUpdateParams,
): Promise<ProjectResponse> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/project/update`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ project_id: projectId, ...params }),
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

export const useUpdateProject = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<
    ProjectResponse,
    Error,
    { projectId: string; params: ProjectUpdateParams }
  >({
    mutationFn: async ({ projectId, params }) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return updateProject(accessToken, projectId, params);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.all });
    },
  });
};
