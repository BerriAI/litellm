"use client";

import { useQuery } from "@tanstack/react-query";
import { keyCreateCall, teamListCall } from "@/components/networking";
import { Team } from "@/components/key_team_helpers/key_list";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";

export const PLAYGROUND_TEAM_KEY_DURATION = "24h";

export const playgroundTeamKeyStorageKey = (teamId: string) => `playgroundTeamKey:${teamId}`;

interface UsePlaygroundTeamSessionArgs {
  accessToken: string | null;
  userID: string | null;
  teamId: string | null;
  enabled: boolean;
}

interface PlaygroundTeamSession {
  teams: Team[];
  teamKey: string | null;
  isLoadingTeams: boolean;
  isMintingKey: boolean;
  error: string | null;
}

const readCachedKey = (teamId: string): string | null => {
  try {
    return getSecureItem(playgroundTeamKeyStorageKey(teamId));
  } catch {
    return null;
  }
};

const cacheKey = (teamId: string, key: string): void => {
  try {
    setSecureItem(playgroundTeamKeyStorageKey(teamId), key);
  } catch {
    return;
  }
};

const errorMessage = (error: unknown, fallback: string): string =>
  error instanceof Error && error.message ? error.message : fallback;

export const usePlaygroundTeamSession = ({
  accessToken,
  userID,
  teamId,
  enabled,
}: UsePlaygroundTeamSessionArgs): PlaygroundTeamSession => {
  const teamsQuery = useQuery({
    queryKey: ["playgroundTeams", userID],
    queryFn: async (): Promise<Team[]> => {
      if (!accessToken) return [];
      const response: unknown = await teamListCall(accessToken, null, userID);
      return Array.isArray(response) ? (response as Team[]) : [];
    },
    enabled: enabled && !!accessToken,
  });

  const teamKeyQuery = useQuery({
    queryKey: ["playgroundTeamKey", teamId],
    queryFn: async (): Promise<string> => {
      if (!accessToken || !userID || !teamId) {
        throw new Error("Sign in and select a team to chat as that team");
      }
      const cached = readCachedKey(teamId);
      if (cached) {
        return cached;
      }
      const response: { key?: string } = await keyCreateCall(accessToken, userID, {
        team_id: teamId,
        duration: PLAYGROUND_TEAM_KEY_DURATION,
        key_alias: `playground-${teamId}-${Date.now()}`,
      });
      if (!response?.key) {
        throw new Error("Creating a key for this team returned no key");
      }
      cacheKey(teamId, response.key);
      return response.key;
    },
    enabled: enabled && !!accessToken && !!userID && !!teamId,
    retry: false,
    staleTime: Infinity,
  });

  let error: string | null = null;
  if (teamsQuery.error) {
    error = errorMessage(teamsQuery.error, "Failed to load teams");
  } else if (teamKeyQuery.error) {
    error = errorMessage(teamKeyQuery.error, "Failed to create a key for this team");
  }

  return {
    teams: teamsQuery.data ?? [],
    teamKey: teamKeyQuery.data ?? null,
    isLoadingTeams: teamsQuery.isFetching,
    isMintingKey: teamKeyQuery.isFetching,
    error,
  };
};
