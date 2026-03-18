import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";
import { all_admin_roles } from "@/utils/roles";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { ProjectResponse, projectKeys } from "./useProjects";

// ── Fetch function ───────────────────────────────────────────────────────────

const fetchProjectDetails = async (
  accessToken: string,
  projectId: string,
): Promise<ProjectResponse> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/project/info?project_id=${encodeURIComponent(projectId)}`;

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

export const useProjectDetails = (projectId?: string) => {
  const { accessToken, userRole } = useAuthorized();
  const queryClient = useQueryClient();

  return useQuery<ProjectResponse>({
    queryKey: projectKeys.detail(projectId!),
    queryFn: async () => fetchProjectDetails(accessToken!, projectId!),
    enabled:
      Boolean(accessToken && projectId) &&
      all_admin_roles.includes(userRole || ""),

    // Seed from the list cache when available
    initialData: () => {
      if (!projectId) return undefined;

      const projects = queryClient.getQueryData<ProjectResponse[]>(
        projectKeys.list({}),
      );

      return projects?.find((p) => p.project_id === projectId);
    },
  });
};
