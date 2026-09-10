import type { ProjectDailySpendRow } from "@/components/networking";

/** One row per project, summed across the whole range: the shape the breakdown table and donut chart render. */
export interface ProjectSpendRow extends Record<string, unknown> {
  project_id: string;
  project_alias: string;
  spend: number;
  requests: number;
  successful_requests: number;
  failed_requests: number;
  tokens: number;
}

/** One point per day, spend summed across every selected project: the shape the daily spend chart renders. */
export interface DailyProjectSpendPoint extends Record<string, unknown> {
  date: string;
  spend: number;
}

export interface ProjectUsageSummary {
  total_spend: number;
  total_api_requests: number;
  total_successful_requests: number;
  total_failed_requests: number;
  total_tokens: number;
}

const EMPTY_SUMMARY: ProjectUsageSummary = {
  total_spend: 0,
  total_api_requests: 0,
  total_successful_requests: 0,
  total_failed_requests: 0,
  total_tokens: 0,
};

/** Aggregate totals across every row, for the summary tiles. */
export const summarizeProjectUsage = (rows: ProjectDailySpendRow[]): ProjectUsageSummary =>
  rows.reduce(
    (totals, row) => ({
      total_spend: totals.total_spend + row.spend,
      total_api_requests: totals.total_api_requests + row.api_requests,
      total_successful_requests: totals.total_successful_requests + row.successful_requests,
      total_failed_requests: totals.total_failed_requests + row.failed_requests,
      total_tokens: totals.total_tokens + row.total_tokens,
    }),
    EMPTY_SUMMARY,
  );

/** One point per day, spend summed across every selected project, sorted oldest first. */
export const buildDailySpendSeries = (rows: ProjectDailySpendRow[]): DailyProjectSpendPoint[] => {
  const spendByDate = new Map<string, number>();
  rows.forEach((row) => {
    spendByDate.set(row.date, (spendByDate.get(row.date) ?? 0) + row.spend);
  });
  return Array.from(spendByDate, ([date, spend]) => ({ date, spend })).sort((a, b) => a.date.localeCompare(b.date));
};

/**
 * One row per project, spend/tokens/requests summed across the whole range, sorted by
 * spend descending like every other "top X" breakdown on the usage page.
 */
export const buildProjectSpendBreakdown = (rows: ProjectDailySpendRow[]): ProjectSpendRow[] => {
  const byProject = new Map<string, ProjectSpendRow>();
  rows.forEach((row) => {
    const existing = byProject.get(row.project_id);
    if (existing) {
      existing.spend += row.spend;
      existing.requests += row.api_requests;
      existing.successful_requests += row.successful_requests;
      existing.failed_requests += row.failed_requests;
      existing.tokens += row.total_tokens;
      return;
    }
    const newRow: ProjectSpendRow = {
      project_id: row.project_id,
      project_alias: row.project_alias || row.project_id,
      spend: row.spend,
      requests: row.api_requests,
      successful_requests: row.successful_requests,
      failed_requests: row.failed_requests,
      tokens: row.total_tokens,
    };
    byProject.set(row.project_id, newRow);
  });
  return Array.from(byProject.values()).sort((a, b) => b.spend - a.spend);
};
