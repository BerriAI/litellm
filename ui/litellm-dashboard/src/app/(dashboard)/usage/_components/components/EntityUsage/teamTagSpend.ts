import type { DailyData } from "@/components/UsagePage/types";

export interface TeamTagSpendRow {
  tag: string;
  spend: number;
  api_requests: number;
  successful_requests: number;
  failed_requests: number;
  total_tokens: number;
}

const emptyRow = (tag: string): TeamTagSpendRow => ({
  tag,
  spend: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  total_tokens: 0,
});

/**
 * The Team view's per-tag breakdown, from a group_by=tag report whose
 * breakdown.entities are keyed by tag: one row per tag summed across days,
 * zero-spend tags dropped, highest spend first (ties by tag name).
 */
export const teamTagSpendRows = (days: readonly DailyData[]): TeamTagSpendRow[] => {
  const byTag = new Map<string, TeamTagSpendRow>();
  for (const day of days) {
    for (const [tag, { metrics }] of Object.entries(day.breakdown.entities ?? {})) {
      const row = byTag.get(tag) ?? emptyRow(tag);
      const summed: TeamTagSpendRow = {
        tag,
        spend: row.spend + metrics.spend,
        api_requests: row.api_requests + metrics.api_requests,
        successful_requests: row.successful_requests + metrics.successful_requests,
        failed_requests: row.failed_requests + metrics.failed_requests,
        total_tokens: row.total_tokens + metrics.total_tokens,
      };
      byTag.set(tag, summed);
    }
  }
  return [...byTag.values()]
    .filter((row) => row.spend > 0)
    .sort((a, b) => b.spend - a.spend || a.tag.localeCompare(b.tag));
};

/** The card answers "what did this team spend per tag", so it needs exactly one team. */
export const singleTeamId = (selectedTeamIds: readonly string[]): string | null =>
  selectedTeamIds.length === 1 ? selectedTeamIds[0] : null;
