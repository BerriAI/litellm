import type { components } from "@/lib/http/schema";

type Activity = components["schemas"]["SpendAnalyticsPaginatedResponse"];
export interface AffectedTeam {
  id: string;
  requests: number;
  keys: number;
  lastUsedDate: string;
}

export function affectedTeams(activity: Activity): AffectedTeam[] {
  const teams = new Map<string, AffectedTeam>();
  for (const day of activity.results ?? []) {
    for (const [id, entry] of Object.entries(day.breakdown?.entities ?? {})) {
      if (entry.metrics.api_requests <= 0) continue;
      const team = teams.get(id) ?? {
        id,
        requests: 0,
        keys: activity.metadata?.entity_total_api_keys?.[id] ?? 0,
        lastUsedDate: "",
      };
      team.requests += entry.metrics.api_requests;
      if (day.date > team.lastUsedDate) team.lastUsedDate = day.date;
      teams.set(id, team);
    }
  }
  return [...teams.values()].sort((a, b) => b.requests - a.requests);
}
