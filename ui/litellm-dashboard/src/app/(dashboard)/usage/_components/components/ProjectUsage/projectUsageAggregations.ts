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
  const spendByDate = rows.reduce<Record<string, number>>(
    (totals, row) => ({ ...totals, [row.date]: (totals[row.date] ?? 0) + row.spend }),
    {},
  );
  return Object.entries(spendByDate)
    .map(([date, spend]) => ({ date, spend }))
    .sort((a, b) => a.date.localeCompare(b.date));
};

const groupByProjectId = (rows: ProjectDailySpendRow[]): Record<string, ProjectDailySpendRow[]> =>
  rows.reduce<Record<string, ProjectDailySpendRow[]>>(
    (groups, row) => ({ ...groups, [row.project_id]: [...(groups[row.project_id] ?? []), row] }),
    {},
  );

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
    { spend: 0, requests: 0, successful_requests: 0, failed_requests: 0, tokens: 0 },
  );
  return { project_id, project_alias: project_alias || project_id, ...totals };
};

const disambiguateAliases = (rows: ProjectSpendRow[]): ProjectSpendRow[] => {
  const aliasCounts = rows.reduce<Record<string, number>>(
    (counts, row) => ({ ...counts, [row.project_alias]: (counts[row.project_alias] ?? 0) + 1 }),
    {},
  );
  return rows.map((row) =>
    aliasCounts[row.project_alias] > 1 ? { ...row, project_alias: `${row.project_alias} (${row.project_id})` } : row,
  );
};

export const buildProjectSpendBreakdown = (rows: ProjectDailySpendRow[]): ProjectSpendRow[] => {
  const summarized = Object.values(groupByProjectId(rows)).map(summarizeProjectGroup);
  return disambiguateAliases(summarized).sort((a, b) => b.spend - a.spend);
};

export const humanizeBackendListMessage = (message: string): string =>
  message.replace(/\[([^\]]*)]\s*$/, (_match, listContents: string) => {
    const items = [...listContents.matchAll(/'([^']*)'|"([^"]*)"/g)].map((m) => m[1] ?? m[2]);
    return items.length > 0 ? items.join(", ") : listContents;
  });
