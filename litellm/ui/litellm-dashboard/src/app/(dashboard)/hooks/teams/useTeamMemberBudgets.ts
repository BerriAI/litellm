"use client";

import { useQueries, type UseQueryResult } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";

import type { TeamMemberBudgetSource } from "@/components/shared/InheritedBudgetHint";
import { tempBudgetExpiryMs } from "@/components/shared/InheritedBudgetHint";
import { teamInfoCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { fetchTeamMembersByUserId, teamKeys } from "@/app/(dashboard)/hooks/teams/useTeams";

export const teamMemberBudgetQueryKey = (teamId: string) => [...teamKeys.all, "memberBudget", teamId] as const;

const MAX_TIMER_DELAY_MS = 2_147_483_647;

const fetchTeamMemberBudgetSource = async (
  accessToken: string,
  teamId: string,
  userIds: readonly string[],
): Promise<TeamMemberBudgetSource> => {
  const [teamInfo, members] = await Promise.all([
    teamInfoCall(accessToken, teamId, { includeMembers: false, keyLimit: 1 }) as Promise<TeamMemberBudgetSource>,
    fetchTeamMembersByUserId(accessToken, teamId, userIds),
  ]);
  return {
    team_info: teamInfo.team_info,
    team_memberships: members.flatMap((member) =>
      member.user_id == null
        ? []
        : [{ user_id: member.user_id, spend: member.spend, litellm_budget_table: member.litellm_budget_table }],
    ),
  };
};

export const useTeamMemberBudgets = (
  userIdsByTeam: Readonly<Record<string, readonly string[]>>,
): Readonly<Record<string, TeamMemberBudgetSource>> => {
  const { accessToken, userId } = useAuthorized();
  const teams = useMemo(
    () => Object.entries(userIdsByTeam).map(([teamId, userIds]) => ({ teamId, userIds: [...new Set(userIds)].sort() })),
    [userIdsByTeam],
  );
  const [lastExpiryCheckMs, setLastExpiryCheckMs] = useState<number | null>(null);
  const combine = useMemo(
    () =>
      (results: UseQueryResult<TeamMemberBudgetSource>[]): Readonly<Record<string, TeamMemberBudgetSource>> =>
        Object.fromEntries(
          results.flatMap((result, index) =>
            result.isSuccess && result.data ? [[teams[index].teamId, result.data]] : [],
          ),
        ),
    [teams],
  );

  const budgets = useQueries({
    queries: teams.map(({ teamId, userIds }) => ({
      queryKey: [...teamMemberBudgetQueryKey(teamId), userId, userIds],
      queryFn: () => fetchTeamMemberBudgetSource(accessToken!, teamId, userIds),
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
