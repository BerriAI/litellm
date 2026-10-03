"use client";

import { useQueries, type UseQueryResult } from "@tanstack/react-query";
import { useMemo } from "react";

import type { TeamMemberBudgetSource } from "@/components/shared/InheritedBudgetHint";
import { teamInfoCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { teamKeys } from "@/app/(dashboard)/hooks/teams/useTeams";

export const teamMemberBudgetQueryKey = (teamId: string) => [...teamKeys.all, "memberBudget", teamId] as const;

export const useTeamMemberBudgets = (teamIds: readonly string[]): Readonly<Record<string, TeamMemberBudgetSource>> => {
  const { accessToken, userId } = useAuthorized();
  const uniqueTeamIds = useMemo(() => [...new Set(teamIds)], [teamIds]);
  const combine = useMemo(
    () =>
      (results: UseQueryResult<TeamMemberBudgetSource>[]): Readonly<Record<string, TeamMemberBudgetSource>> =>
        Object.fromEntries(
          results.flatMap((result, index) =>
            result.isSuccess && result.data ? [[uniqueTeamIds[index], result.data]] : [],
          ),
        ),
    [uniqueTeamIds],
  );

  return useQueries({
    queries: uniqueTeamIds.map((teamId) => ({
      queryKey: [...teamMemberBudgetQueryKey(teamId), userId],
      queryFn: async (): Promise<TeamMemberBudgetSource> =>
        (await teamInfoCall(accessToken!, teamId, { keyLimit: 1 })) as TeamMemberBudgetSource,
      enabled: Boolean(accessToken && userId),
      retry: false,
    })),
    combine,
  });
};
