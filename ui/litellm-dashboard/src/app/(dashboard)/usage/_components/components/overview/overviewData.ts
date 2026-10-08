import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import { stackedUsageColor } from "@/components/shared/charts";
import { EMPTY_DAILY_ACTIVITY_METADATA, type DailyActivityMetadata } from "@/components/UsagePage/dailyActivityApi";
import type { GatewayActivity } from "../gatewayActivity";

export type BreakdownDimension = "models" | "model_groups" | "providers";

/** Extends Record so it satisfies the chart component's row constraint. */
interface RankedRow extends Record<string, unknown> {
  key: string;
  spend: number;
  requests: number;
  successful_requests: number;
  failed_requests: number;
  tokens: number;
}

const EMPTY_RANKED = { spend: 0, requests: 0, successful_requests: 0, failed_requests: 0, tokens: 0 } as const;

/** Sums one breakdown dimension across every day, highest spend first. */
export const rollUpBreakdown = (results: readonly DailyData[], dimension: BreakdownDimension): RankedRow[] => {
  const entries = results.flatMap((day) => Object.entries(day.breakdown[dimension] ?? {}));
  const totals = entries.reduce<ReadonlyMap<string, RankedRow>>((acc, [key, entry]) => {
    const row = acc.get(key) ?? { key, ...EMPTY_RANKED };
    return new Map([
      ...acc,
      [
        key,
        {
          ...row,
          spend: row.spend + entry.metrics.spend,
          requests: row.requests + entry.metrics.api_requests,
          successful_requests: row.successful_requests + (entry.metrics.successful_requests || 0),
          failed_requests: row.failed_requests + (entry.metrics.failed_requests || 0),
          tokens: row.tokens + entry.metrics.total_tokens,
        },
      ],
    ]);
  }, new Map());
  return [...totals.values()].sort((a, b) => b.spend - a.spend);
};

export const OTHER_COLOR = "#94a3b8";
const OTHER_SERIES = "Other";

export interface SeriesDay extends Record<string, unknown> {
  /** ISO date, the chart's category key: unique across years, unlike its display label. */
  date: string;
  label: string;
}

/**
 * `keys` are internal (`s0`, `s1`, ...) so a series named `date`, `label` or `Other` cannot
 * collide with the day's own fields; `labels` carry the display names in the same order.
 */
export interface Series {
  data: SeriesDay[];
  keys: string[];
  labels: string[];
  colors: string[];
}

const seriesKey = (index: number) => `s${index}`;

const formatDayLabel = (date: string): string => {
  const parsed = new Date(`${date}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return date;
  return parsed.toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

const sortByDate = (results: readonly DailyData[]) =>
  [...results].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

export type UsageMetric = "spend" | "tokens" | "requests";

const metricOf = (metrics: SpendMetrics, metric: UsageMetric): number => {
  if (metric === "spend") return metrics.spend;
  if (metric === "tokens") return metrics.total_tokens;
  return metrics.api_requests;
};

export const rankValue = (row: RankedRow, metric: UsageMetric): number => {
  if (metric === "spend") return row.spend;
  if (metric === "tokens") return row.tokens;
  return row.requests;
};

/**
 * Daily `metric` stacked by the top `top` keys of a dimension, the rest folded
 * into "Other" so every bar still sums to the day's total.
 */
export const seriesBy = (
  results: readonly DailyData[],
  dimension: BreakdownDimension,
  metric: UsageMetric,
  top = 6,
): Series => {
  const leaders = rollUpBreakdown(results, dimension)
    .filter((row) => rankValue(row, metric) > 0)
    .sort((a, b) => rankValue(b, metric) - rankValue(a, metric))
    .slice(0, top)
    .map((row) => row.key);
  const days = sortByDate(results).map((day) => {
    const values = leaders.map((leader) => {
      const entry = day.breakdown[dimension]?.[leader];
      return entry ? metricOf(entry.metrics, metric) : 0;
    });
    const named = values.reduce((sum, value) => sum + value, 0);
    // Rounding in the per-key sums leaves dust; only a real remainder earns a segment.
    const remainder = metricOf(day.metrics, metric) - named;
    return { day, values, other: remainder > 1e-6 ? remainder : 0 };
  });
  const hasOther = days.some(({ other }) => other > 0) || leaders.length === 0;
  const labels = hasOther ? [...leaders, OTHER_SERIES] : leaders;
  const keys = labels.map((_, index) => seriesKey(index));
  const data = days.map(({ day, values, other }) => ({
    date: day.date,
    label: formatDayLabel(day.date),
    ...Object.fromEntries((hasOther ? [...values, other] : values).map((value, index) => [keys[index], value])),
  }));
  const colors = labels.map((label, index) =>
    hasOther && index === labels.length - 1 ? OTHER_COLOR : stackedUsageColor(index),
  );
  return { data, keys, labels, colors };
};

export type Granularity = "day" | "week";

const DAY_MS = 86_400_000;

/** Sums a daily series into 7-day buckets anchored on the first day, keeping one point per bucket. */
export const bucketSeries = (series: Series, granularity: Granularity): Series => {
  if (granularity === "day" || series.data.length === 0) return series;
  const origin = Date.parse(`${series.data[0].date}T00:00:00Z`);
  const bucketStart = (date: string) => {
    const index = Math.floor((Date.parse(`${date}T00:00:00Z`) - origin) / (7 * DAY_MS));
    return new Date(origin + index * 7 * DAY_MS).toISOString().slice(0, 10);
  };
  const starts = [...new Set(series.data.map((day) => bucketStart(day.date)))];
  const data = starts.map((start) => {
    const members = series.data.filter((day) => bucketStart(day.date) === start);
    return {
      date: start,
      label: `Wk of ${formatDayLabel(start)}`,
      ...Object.fromEntries(
        series.keys.map((key) => [key, members.reduce((sum, day) => sum + Number(day[key] ?? 0), 0)]),
      ),
    };
  });
  return { ...series, data };
};

/** Bar totals keyed by ISO date, so two years' "Jan 1" in one range never overwrite each other. */
export const bucketTotals = (series: Series): ReadonlyMap<string, number> =>
  new Map(series.data.map((day) => [day.date, series.keys.reduce((sum, key) => sum + Number(day[key] ?? 0), 0)]));

/** Display label for a bucket's ISO date, for chart ticks and tooltip headers. */
export const labelForDate = (series: Series, date: string): string =>
  series.data.find((day) => day.date === date)?.label ?? date;

export interface LeaderRow extends RankedRow {
  share: number;
  /** Share in the later half of the range minus share in the earlier half, in points. */
  delta: number | null;
}

const shareIn = (rows: readonly RankedRow[], metric: UsageMetric) => {
  const total = rows.reduce((sum, row) => sum + rankValue(row, metric), 0);
  const byKey = new Map(rows.map((row) => [row.key, rankValue(row, metric)]));
  return { total, of: (key: string) => (total > 0 ? ((byKey.get(key) ?? 0) / total) * 100 : 0) };
};

/** Share of `metric` per key, plus how that share moved between the two halves of the range. */
export const leaderboard = (
  results: readonly DailyData[],
  dimension: BreakdownDimension,
  metric: UsageMetric,
): LeaderRow[] => {
  const days = sortByDate(results);
  const rows = rollUpBreakdown(days, dimension).sort((a, b) => rankValue(b, metric) - rankValue(a, metric));
  const grand = rows.reduce((sum, row) => sum + rankValue(row, metric), 0);
  const half = Math.floor(days.length / 2);
  const earlier = shareIn(rollUpBreakdown(days.slice(0, half), dimension), metric);
  const later = shareIn(rollUpBreakdown(days.slice(half), dimension), metric);
  const comparable = half > 0 && earlier.total > 0 && later.total > 0;
  return rows
    .filter((row) => rankValue(row, metric) > 0)
    .map((row) => ({
      ...row,
      share: grand > 0 ? (rankValue(row, metric) / grand) * 100 : 0,
      delta: comparable ? later.of(row.key) - earlier.of(row.key) : null,
    }));
};

export const formatMetricValue = (value: number, metric: UsageMetric): string =>
  metric === "spend" ? formatCompactUsd(value) : formatCompact(value);

/** One point per day with the headline totals, for sparklines. */
export const dailyTotals = (results: readonly DailyData[]): SeriesDay[] =>
  sortByDate(results).map((day) => ({
    date: day.date,
    label: formatDayLabel(day.date),
    spend: day.metrics.spend,
    requests: day.metrics.api_requests,
    tokens: day.metrics.total_tokens,
  }));

export const formatUsd = (value: number, digits = 2): string =>
  `$${value.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;

export const formatCompactUsd = (value: number): string => {
  if (value === 0) return "$0";
  if (Math.abs(value) < 100) return `$${value.toFixed(2)}`;
  if (Math.abs(value) < 1000) return `$${value.toFixed(0)}`;
  return `$${Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value)}`;
};

export const formatCompact = (value: number): string =>
  Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);

export const formatLatency = (ms: number): string => (ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`);

export interface OverviewTotals {
  spend: number;
  requests: number;
  successful: number;
  failed: number;
  tokens: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  avgCostPerRequest: number;
  avgLatencyMs: number | null;
  successRate: number | null;
}

/**
 * Request counts prefer the gateway's own tallies, falling back to the spend
 * aggregate, the same precedence the request tiles have always used.
 */
export const overviewTotals = (raw: DailyActivityMetadata, gateway: GatewayActivity | null): OverviewTotals => {
  // The schema marks every total optional; filling the defaults once keeps the arithmetic below plain.
  const metadata = { ...EMPTY_DAILY_ACTIVITY_METADATA, ...raw };
  const successful = gateway?.total_successful_requests ?? metadata.total_successful_requests;
  const failed = gateway?.total_failed_requests ?? metadata.total_failed_requests;
  const timed = metadata.total_timed_requests;
  return {
    spend: metadata.total_spend,
    requests: gateway ? successful + failed : metadata.total_api_requests,
    successful,
    failed,
    tokens: metadata.total_tokens,
    inputTokens: metadata.total_prompt_tokens,
    outputTokens: metadata.total_completion_tokens,
    cacheReadTokens: metadata.total_cache_read_input_tokens,
    cacheWriteTokens: metadata.total_cache_creation_input_tokens,
    avgCostPerRequest: metadata.total_spend / (metadata.total_api_requests || 1),
    avgLatencyMs: timed > 0 ? metadata.total_response_time_ms / timed : null,
    successRate: successful + failed > 0 ? (successful / (successful + failed)) * 100 : null,
  };
};
