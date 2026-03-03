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

export interface ProjectBudget {
  budget_id: string;
  max_budget: number | null;
  soft_budget: number | null;
  max_parallel_requests: number | null;
  tpm_limit: number | null;
  rpm_limit: number | null;
  model_max_budget: Record<string, number> | null;
  budget_duration: string | null;
}

export interface ProjectResponse {
  project_id: string;
  project_alias: string | null;
  description: string | null;
  team_id: string | null;
  budget_id: string | null;
  metadata: Record<string, unknown> | null;
  models: string[];
  spend: number;
  model_spend: Record<string, number> | null;
  model_rpm_limit: Record<string, number> | null;
  model_tpm_limit: Record<string, number> | null;
  blocked: boolean;
  object_permission_id: string | null;
  created_at: string;
  created_by: string;
  updated_at: string;
  updated_by: string;
  litellm_budget_table: ProjectBudget | null;
}

// ── Query keys (shared across project hooks) ─────────────────────────────────

export const projectKeys = createQueryKeys("projects");

// ── Fetch function ───────────────────────────────────────────────────────────

const fetchProjects = async (
  accessToken: string,
): Promise<ProjectResponse[]> => {
  const baseUrl = getProxyBaseUrl();
  const url = `${baseUrl}/project/list`;

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

export const useProjects = () => {
  const { accessToken, userRole } = useAuthorized();

  return useQuery<ProjectResponse[]>({
    queryKey: projectKeys.list({}),
    queryFn: async () => fetchProjects(accessToken!),
    enabled:
      Boolean(accessToken) && all_admin_roles.includes(userRole || ""),
  });
};
