import { useQuery, UseQueryResult } from "@tanstack/react-query";
import {
  deriveErrorMessage,
  getGlobalLitellmHeaderName,
  getProxyBaseUrl,
} from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export interface TeamMemberInfo {
  user_id: string;
  team_id: string;
  team_alias?: string | null;
  role?: string | null;
  user_email?: string | null;
  budget_id?: string | null;
  spend?: number | null;
  total_spend?: number | null;
  litellm_budget_table?: {
    budget_id?: string;
    soft_budget?: number | null;
    max_budget?: number | null;
    max_parallel_requests?: number | null;
    tpm_limit?: number | null;
    rpm_limit?: number | null;
    model_max_budget?: Record<string, number> | null;
    budget_duration?: string | null;
    budget_reset_at?: string | null;
    allowed_models?: string[] | null;
  } | null;
}

const fetchMyTeamMember = async (
  accessToken: string,
  teamId: string,
): Promise<TeamMemberInfo | null> => {
  const proxyBaseUrl = getProxyBaseUrl();
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/team/${encodeURIComponent(teamId)}/members/me`
    : `/team/${encodeURIComponent(teamId)}/members/me`;

  const response = await fetch(url, {
    method: "GET",
    headers: {
      [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
  });

  // 404 means the caller is not a team member (e.g. proxy admin viewing
  // a team they don't belong to). The "My User" tab is always visible so
  // the fetch fires regardless — return null and let the UI render the
  // empty state instead of a noisy error.
  if (response.status === 404) {
    return null;
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(deriveErrorMessage(errorData));
  }

  return (await response.json()) as TeamMemberInfo;
};

export const useMyTeamMember = (
  teamId: string | null | undefined,
): UseQueryResult<TeamMemberInfo | null> => {
  const { accessToken } = useAuthorized();
  return useQuery<TeamMemberInfo | null>({
    queryKey: ["team", teamId, "members", "me"],
    queryFn: () => fetchMyTeamMember(accessToken!, teamId!),
    enabled: Boolean(accessToken && teamId),
  });
};
