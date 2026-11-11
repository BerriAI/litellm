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

export interface DailyModelRequests extends Record<string, unknown> {
  date: string;
  requests: number;
}

/** Include quiet days so a completed migration remains visible as zero traffic. */
export function dailyModelRequests(
  days: readonly components["schemas"]["DailySpendData"][],
  range: { start_date: string; end_date: string },
): DailyModelRequests[] {
  const counts = new Map<string, number>();
  for (const day of days) counts.set(day.date, (counts.get(day.date) ?? 0) + day.metrics.api_requests);
  const result: DailyModelRequests[] = [];
  const end = new Date(`${range.end_date}T00:00:00Z`);
  for (const day = new Date(`${range.start_date}T00:00:00Z`); day <= end; day.setUTCDate(day.getUTCDate() + 1)) {
    const date = day.toISOString().slice(0, 10);
    result.push({ date, requests: counts.get(date) ?? 0 });
  }
  return result;
}
