import type { DailyActivityAggregatedResponse } from "@/components/UsagePage/dailyActivityApi";

export interface CustomerTeamSpend {
  teamId: string;
  teamAlias: string;
  spend: number;
  requests: number;
  tokens: number;
}

export function customerTeamSpend(response: DailyActivityAggregatedResponse): CustomerTeamSpend[] {
  const entries = (response.results ?? []).flatMap((day) => Object.entries(day.breakdown?.teams ?? {}));
  const ids = [...new Set(entries.map(([id]) => id))];
  return ids
    .map((teamId) => {
      const buckets = entries.filter(([id]) => id === teamId).map(([, entry]) => entry);
      const alias = buckets.find((entry) => typeof entry.metadata?.team_alias === "string")?.metadata?.team_alias;
      return {
        teamId,
        teamAlias: typeof alias === "string" ? alias : teamId,
        spend: buckets.reduce((total, entry) => total + (entry.metrics.spend ?? 0), 0),
        requests: buckets.reduce((total, entry) => total + (entry.metrics.api_requests ?? 0), 0),
        tokens: buckets.reduce((total, entry) => total + (entry.metrics.total_tokens ?? 0), 0),
      };
    })
    .sort((a, b) => b.spend - a.spend || a.teamId.localeCompare(b.teamId));
}
