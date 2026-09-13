import type { ProjectDailySpendRow } from "@/components/networking";

export interface ProjectSpendRow extends Record<string, unknown> {
  project_id: string;
  project_alias: string;
  spend: number;
  requests: number;
  successful_requests: number;
  failed_requests: number;
  tokens: number;
}

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

export const buildDailySpendSeries = (rows: ProjectDailySpendRow[]): DailyProjectSpendPoint[] => {
  const spendByDate = new Map<string, number>();
  for (const row of rows) {
    spendByDate.set(row.date, (spendByDate.get(row.date) ?? 0) + row.spend);
  }
  return [...spendByDate.entries()]
    .map(([date, spend]) => ({ date, spend }))
    .sort((a, b) => a.date.localeCompare(b.date));
};

const groupByProjectId = (rows: ProjectDailySpendRow[]): ProjectDailySpendRow[][] => {
  const groups = new Map<string, ProjectDailySpendRow[]>();
  for (const row of rows) {
    const existing = groups.get(row.project_id);
    if (existing) {
      existing.push(row);
    } else {
      groups.set(row.project_id, [row]);
    }
  }
  return [...groups.values()];
};

const EMPTY_PROJECT_GROUP_TOTALS = {
  spend: 0,
  requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  tokens: 0,
};

const summarizeProjectGroup = (rows: ProjectDailySpendRow[]): ProjectSpendRow => {
  const [{ project_id, project_alias }] = rows;
  const totals = rows.reduce(
    (acc, row) => ({
      spend: acc.spend + row.spend,
      requests: acc.requests + row.api_requests,
      successful_requests: acc.successful_requests + row.successful_requests,
      failed_requests: acc.failed_requests + row.failed_requests,
      tokens: acc.tokens + row.total_tokens,
    }),
    EMPTY_PROJECT_GROUP_TOTALS,
  );
  return { project_id, project_alias: project_alias || project_id, ...totals };
};

const disambiguateAliases = (rows: ProjectSpendRow[]): ProjectSpendRow[] => {
  const aliasCounts = new Map<string, number>();
  for (const row of rows) {
    aliasCounts.set(row.project_alias, (aliasCounts.get(row.project_alias) ?? 0) + 1);
  }
  return rows.map((row) =>
    (aliasCounts.get(row.project_alias) ?? 0) > 1
      ? { ...row, project_alias: `${row.project_alias} (${row.project_id})` }
      : row,
  );
};

export const buildProjectSpendBreakdown = (rows: ProjectDailySpendRow[]): ProjectSpendRow[] => {
  const summarized = groupByProjectId(rows).map(summarizeProjectGroup);
  return disambiguateAliases(summarized).sort((a, b) => b.spend - a.spend);
};
