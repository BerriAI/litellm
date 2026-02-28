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

export interface ProjectCreateParams {
  project_alias?: string;
  description?: string;
  team_id: string;
  models?: string[];
  max_budget?: number;
  blocked?: boolean;
  metadata?: Record<string, unknown>;
  model_rpm_limit?: Record<string, number>;
  model_tpm_limit?: Record<string, number>;
}

// ── Fetch function ───────────────────────────────────────────────────────────

const createProject = async (
  accessToken: string,
  params: ProjectCreateParams,
): Promise<ProjectResponse> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/project/new`;

  const response = await fetch(url, {
    method: "POST",
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

export const useCreateProject = () => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();

  return useMutation<ProjectResponse, Error, ProjectCreateParams>({
    mutationFn: async (params) => {
      if (!accessToken) {
        throw new Error("Access token is required");
      }
      return createProject(accessToken, params);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: projectKeys.all });
    },
  });
};
