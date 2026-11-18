import {
  formatDayLabel,
  rankValue,
  rollUpBreakdown,
  seriesBy,
  type Series,
  type UsageMetric,
} from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { stackedUsageColor } from "@/components/shared/charts";
import type { DailyData } from "@/components/UsagePage/types";
import i18n from "@/i18n/i18n";
import type { CustomerHierarchyUsage } from "./customerUsageApi";
import type { CustomerUsageDimension } from "./CustomerUsageRankings";
import { customerModelBreakdown } from "./customerUsageTotals";

const TOP_SERIES = 8;

const CACHE_TOKEN_SERIES = [
  { key: "cacheRead", label: "customerPortal.tokenUsage.cacheReadTokens", color: "#3b82f6" },
  { key: "cacheWrite", label: "customerPortal.tokenUsage.cacheWriteTokens", color: "#a855f7" },
  { key: "other", label: "customerPortal.tokenUsage.otherTokens", color: "#cbd5e1" },
] as const;
const UNCLASSIFIED_TOKEN_SERIES = {
  key: "unclassified",
  label: "customerPortal.tokenUsage.unclassifiedTokens",
  color: "#64748b",
} as const;

export const customerCacheTokenSeries = (days: readonly DailyData[]): Series => {
  const data = days.map(({ date, metrics }) => {
    const cacheRead = metrics.cache_read_input_tokens ?? 0;
    const cacheWrite = metrics.cache_creation_input_tokens ?? 0;
    const total = metrics.total_tokens;
    const point = { date, label: formatDayLabel(date) };
    if (cacheRead + cacheWrite > Math.min(metrics.prompt_tokens, total)) {
      return { ...point, cacheRead: 0, cacheWrite: 0, other: 0, unclassified: total };
    }
    return { ...point, cacheRead, cacheWrite, other: total - cacheRead - cacheWrite, unclassified: 0 };
  });
  const segments = data.some(({ unclassified }) => unclassified > 0)
    ? [...CACHE_TOKEN_SERIES, UNCLASSIFIED_TOKEN_SERIES]
    : CACHE_TOKEN_SERIES;
  return {
    data,
    keys: segments.map(({ key }) => key),
    labels: segments.map(({ label }) => i18n.t(label)),
    colors: segments.map(({ color }) => color),
  };
};

const entityDays = (
  primary: readonly DailyData[],
  entities: readonly DailyData[],
  labels: ReadonlyMap<string, string>,
): DailyData[] => {
  const byDate = new Map(entities.map((day) => [day.date, day]));
  return primary.map((day) => ({
    ...day,
    breakdown: {
      ...day.breakdown,
      model_groups: Object.fromEntries(
        Object.entries(byDate.get(day.date)?.breakdown.entities ?? {}).filter(([id]) => id && labels.has(id)),
      ),
    },
  }));
};

export const customerUsageSeries = (
  data: CustomerHierarchyUsage,
  dimension: CustomerUsageDimension,
  metric: UsageMetric,
  colorByModel: ReadonlyMap<string, string>,
): Series => {
  if (dimension === "models") {
    const days = data.results.map((day) => ({
      ...day,
      breakdown: {
        ...day.breakdown,
        model_groups: customerModelBreakdown(day),
      },
    }));
    return seriesBy(days, "model_groups", metric, { top: TOP_SERIES, colorByKey: colorByModel });
  }

  const unnamedKey = i18n.t("customerPortal.analytics.unnamedKey");
  const rows = data[dimension];
  const labels = new Map(
    rows
      .filter((row) => row.id)
      .map((row) => {
        const label = row.label.trim();
        return [row.id, dimension === "api_keys" && (!label || label === row.id) ? unnamedKey : label];
      }),
  );
  const days = dimension === "api_keys" ? data.results : entityDays(data.results, data.teamResults, labels);
  const breakdown = dimension === "api_keys" ? "api_keys" : "model_groups";
  const ranking = rollUpBreakdown(days, breakdown).sort((a, b) => b.spend - a.spend || a.key.localeCompare(b.key));
  const colors = new Map(ranking.map((row, index) => [row.key, stackedUsageColor(index)]));
  const series = seriesBy(days, breakdown, metric, { top: TOP_SERIES, colorByKey: colors });
  const namedCount = Math.min(TOP_SERIES, ranking.filter((row) => rankValue(row, metric) > 0).length);
  return {
    ...series,
    labels: series.labels.map((id, index) => (index < namedCount ? labels.get(id) ?? unnamedKey : id)),
  };
};
