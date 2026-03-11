import { keepPreviousData, useQuery, useQueryClient, UseQueryResult } from "@tanstack/react-query";
import { Team } from "@/components/key_team_helpers/key_list";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchTeams } from "@/app/(dashboard)/networking";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";
import { teamInfoCall } from "@/components/networking";
import {
  getProxyBaseUrl,
  getGlobalLitellmHeaderName,
  deriveErrorMessage,
  handleError,
} from "@/components/networking";

export interface TeamsResponse {
  teams: Team[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface DeletedTeam extends Team {
  deleted_at: string;
  deleted_by: string;
}


export interface TeamListCallOptions {
  organizationID?: string | null;
  teamID?: string | null;
  team_alias?: string | null;
  userID?: string | null;
  sortBy?: string | null;
  sortOrder?: string | null;
  status?: string | null;
}

export const teamListCall = async (
  accessToken: string,
  page: number,
  pageSize: number,
  options: TeamListCallOptions = {},
) => {
  /**
   * Get all available teams on proxy
   */
  try {
    const baseUrl = getProxyBaseUrl();
    
    const params = new URLSearchParams(
      Object.entries({
        team_id: options.teamID,
        organization_id: options.organizationID,
        team_alias: options.team_alias,
        user_id: options.userID,
        page,
        page_size: pageSize,
        sort_by: options.sortBy,
        sort_order: options.sortOrder,
        status: options.status,
      })
        .filter(([, value]) => value !== undefined && value !== null)
        .map(([key, value]) => [key, String(value)]),
    );

    const url = `${baseUrl ? `${baseUrl}/v2/team/list` : "/v2/team/list"}?${params}`;

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

    const data = await response.json();
    console.log("/v2/team/list API Response:", data);
    return data;
  } catch (error) {
    console.error("Failed to list teams:", error);
    throw error;
  }
};

const teamKeys = createQueryKeys("teams");
export const useTeams = (): UseQueryResult<Team[]> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<Team[]>({
    queryKey: teamKeys.list({}),
    queryFn: async () => await fetchTeams(accessToken!, userId, userRole, null),
    enabled: Boolean(accessToken),
  });
};

export const useTeam = (teamId?: string) => {
  const { accessToken } = useAuthorized();
  const queryClient = useQueryClient();
  return useQuery<Team>({
    queryKey: teamKeys.detail(teamId!),
    enabled: Boolean(accessToken && teamId),

    queryFn: async () => {
      if (!accessToken || !teamId) {
        throw new Error("Missing auth or teamId");
      }

      return teamInfoCall(accessToken, teamId);
    },

    initialData: () => {
      if (!teamId) return undefined;

      const teams = queryClient.getQueryData<Team[]>(teamKeys.list({}));

      return teams?.find((team) => team.team_id === teamId);
    },
  });
};

const deletedTeamListCall = async (
  accessToken: string,
  page: number,
  pageSize: number,
  options: TeamListCallOptions = {},
) => {
  /**
   * Get deleted teams from proxy
   */
  try {
    const baseUrl = getProxyBaseUrl();
    
    const params = new URLSearchParams(
      Object.entries({
        team_id: options.teamID,
        organization_id: options.organizationID,
        team_alias: options.team_alias,
        user_id: options.userID,
        page,
        page_size: pageSize,
        sort_by: options.sortBy,
        sort_order: options.sortOrder,
        status: "deleted",
      })
        .filter(([, value]) => value !== undefined && value !== null)
        .map(([key, value]) => [key, String(value)]),
    );

    const url = `${baseUrl ? `${baseUrl}/v2/team/list` : "/v2/team/list"}?${params}`;

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

    const data = await response.json();
    console.log("/team/list?status=deleted API Response:", data);
    
    // Extract teams array from response if it's wrapped in a response object
    // Otherwise return the data directly if it's already an array
    if (data && typeof data === 'object' && 'teams' in data) {
      return data.teams as DeletedTeam[];
    }
    return data as DeletedTeam[];
  } catch (error) {
    console.error("Failed to list deleted teams:", error);
    throw error;
  }
};

export const deletedTeamKeys = createQueryKeys("deletedTeams");
export const useDeletedTeams = (
  page: number,
  pageSize: number,
  options: TeamListCallOptions = {},
): UseQueryResult<DeletedTeam[]> => {
  const { accessToken } = useAuthorized();

  return useQuery<DeletedTeam[]>({
    queryKey: deletedTeamKeys.list({ page, limit: pageSize, ...options }),
    queryFn: async () => await deletedTeamListCall(accessToken!, page, pageSize, options),
    enabled: Boolean(accessToken),
    staleTime: 30000, // 30 seconds
    placeholderData: keepPreviousData,
  });
};