"use client";

import { useQueries, type UseQueryResult } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import type { TeamMemberBudgetSource } from "@/components/shared/InheritedBudgetHint";
import { tempBudgetExpiryMs } from "@/components/shared/InheritedBudgetHint";
import { teamInfoCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { teamKeys } from "@/app/(dashboard)/hooks/teams/useTeams";

export const teamMemberBudgetQueryKey = (teamId: string) => [...teamKeys.all, "memberBudget", teamId] as const;

const MAX_TIMER_DELAY_MS = 2_147_483_647;

export const useTeamMemberBudgets = (teamIds: readonly string[]): Readonly<Record<string, TeamMemberBudgetSource>> => {
  const { accessToken, userId } = useAuthorized();
  const uniqueTeamIds = useMemo(() => [...new Set(teamIds)], [teamIds]);
  const [lastExpiryCheckMs, setLastExpiryCheckMs] = useState<number | null>(null);
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

  const budgets = useQueries({
    queries: uniqueTeamIds.map((teamId) => ({
      queryKey: [...teamMemberBudgetQueryKey(teamId), userId],
      queryFn: async (): Promise<TeamMemberBudgetSource> =>
        (await teamInfoCall(accessToken!, teamId, { keyLimit: 1 })) as TeamMemberBudgetSource,
      enabled: Boolean(accessToken && userId),
      retry: false,
    })),
    combine,
  });

  const pendingExpiries = useMemo(
    () =>
      Object.values(budgets).flatMap((teamInfo) =>
        (teamInfo.team_memberships ?? [])
          .map((membership) => tempBudgetExpiryMs(membership.litellm_budget_table))
          .filter((expiry): expiry is number => expiry !== null),
      ),
    [budgets],
  );

  useEffect(() => {
    const nextExpiryMs = pendingExpiries.filter((expiry) => expiry > Date.now()).sort((a, b) => a - b)[0];
    if (nextExpiryMs === undefined) return;
    const timer = setTimeout(
      () => setLastExpiryCheckMs(Date.now()),
      Math.min(nextExpiryMs - Date.now() + 1, MAX_TIMER_DELAY_MS),
    );
    return () => clearTimeout(timer);
  }, [pendingExpiries, lastExpiryCheckMs]);

  return useMemo(() => (lastExpiryCheckMs === null ? budgets : { ...budgets }), [budgets, lastExpiryCheckMs]);
};
