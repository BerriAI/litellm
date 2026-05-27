import type { DateRangePickerValue } from "@tremor/react";

import type {
  BreakdownMetrics,
  DailyData,
  KeyMetricWithMetadata,
  MetricWithMetadata,
  SpendMetrics,
} from "../types";

/**
 * Label used on the X-axis when the Daily Spend chart is collapsed to a single
 * bar because the selected date range covers only one local calendar day.
 *
 * The Daily Spend bar chart aggregates rows returned by `/user/daily/activity`,
 * which are keyed by date. When the selected range is "Today", the backend can
 * return rows that span two adjacent calendar dates due to UTC vs local-time
 * boundaries — producing two bars labeled with consecutive dates for what the
 * user thinks of as one day. We avoid that by collapsing every row into a
 * single bar with this label.
 */
export const SINGLE_DAY_TIME_LABEL = "12 AM";

const ZERO_METRICS: SpendMetrics = {
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
};

function addMetrics(target: SpendMetrics, src: Partial<SpendMetrics> | undefined): void {
  if (!src) return;
  target.spend += src.spend ?? 0;
  target.prompt_tokens += src.prompt_tokens ?? 0;
  target.completion_tokens += src.completion_tokens ?? 0;
  target.total_tokens += src.total_tokens ?? 0;
  target.api_requests += src.api_requests ?? 0;
  target.successful_requests += src.successful_requests ?? 0;
  target.failed_requests += src.failed_requests ?? 0;
  target.cache_read_input_tokens += src.cache_read_input_tokens ?? 0;
  target.cache_creation_input_tokens += src.cache_creation_input_tokens ?? 0;
}

/**
 * Merge a source {key -> MetricWithMetadata} map into `target`, summing
 * `metrics` and `api_key_breakdown` for matching keys and preserving the
 * earliest-seen `metadata`.
 */
function mergeMetricMap(
  target: { [k: string]: MetricWithMetadata },
  src: { [k: string]: MetricWithMetadata } | undefined,
): void {
  if (!src) return;
  for (const [k, v] of Object.entries(src)) {
    if (!target[k]) {
      target[k] = {
        metrics: { ...ZERO_METRICS },
        metadata: v.metadata ?? {},
        api_key_breakdown: {},
      };
    }
    addMetrics(target[k].metrics, v.metrics);
    mergeKeyMap(target[k].api_key_breakdown, v.api_key_breakdown);
  }
}

function mergeKeyMap(
  target: { [k: string]: KeyMetricWithMetadata },
  src: { [k: string]: KeyMetricWithMetadata } | undefined,
): void {
  if (!src) return;
  for (const [k, v] of Object.entries(src)) {
    if (!target[k]) {
      target[k] = {
        metrics: { ...ZERO_METRICS },
        metadata: v.metadata ?? { key_alias: null, team_id: null, tags: [] },
      };
    }
    addMetrics(target[k].metrics, v.metrics);
  }
}

function emptyBreakdown(): BreakdownMetrics {
  return {
    models: {},
    model_groups: {},
    mcp_servers: {},
    providers: {},
    api_keys: {},
    entities: {},
  };
}

/**
 * True when `dateValue` represents a single local calendar day (from and to
 * are on the same Y-M-D in the browser timezone). False if either bound is
 * missing or if the bounds span more than one calendar day.
 */
export function isSingleDayRange(dateValue: DateRangePickerValue | null | undefined): boolean {
  if (!dateValue) return false;
  const { from, to } = dateValue;
  if (!from || !to) return false;
  return (
    from.getFullYear() === to.getFullYear() &&
    from.getMonth() === to.getMonth() &&
    from.getDate() === to.getDate()
  );
}

/**
 * Sum all metric fields and merge all breakdown maps across `rows` into a
 * single `DailyData` row whose `date` field is `label`. Used to render one bar
 * instead of N bars when the selected range is a single calendar day. Per-row
 * `breakdown.{models, providers, api_keys, entities, ...}` maps are deep-merged
 * so the existing tooltip "Spend by Entity"/"Spend by Model" lists keep all
 * source-row contributions instead of silently showing only the first row.
 */
export function collapseDailyResults<T extends DailyData>(rows: T[], label: string): T {
  const metrics: SpendMetrics = { ...ZERO_METRICS };
  const breakdown: BreakdownMetrics = emptyBreakdown();
  for (const r of rows) {
    addMetrics(metrics, r.metrics);
    const b = r.breakdown;
    if (b) {
      mergeMetricMap(breakdown.models, b.models);
      mergeMetricMap(breakdown.model_groups, b.model_groups);
      mergeMetricMap(breakdown.mcp_servers, b.mcp_servers);
      mergeMetricMap(breakdown.providers, b.providers);
      mergeKeyMap(breakdown.api_keys, b.api_keys);
      mergeMetricMap(breakdown.entities, b.entities);
      if (b.endpoints) {
        if (!breakdown.endpoints) breakdown.endpoints = {};
        mergeMetricMap(breakdown.endpoints, b.endpoints);
      }
    }
  }
  return {
    ...(rows[0] ?? {}),
    date: label,
    metrics,
    breakdown,
  } as T;
}

/**
 * Prepare the data array for the Daily Spend bar chart.
 *
 *   - Multi-day range: returns a copy of `rows` sorted ascending by date.
 *     (The input array is never mutated.)
 *   - Single-day range: returns one element — a `DailyData` row whose `date`
 *     is `SINGLE_DAY_TIME_LABEL` and whose metrics + breakdown are the
 *     deep-merged sum of every row in the range. Eliminates the "multiple
 *     bars all labeled today" bug reported in LIT-3383 while preserving the
 *     per-entity / per-model tooltip data shown by `EntityUsage`.
 *   - Empty input: returns `[]` for both ranges.
 */
export function getDailySpendChartData<T extends DailyData>(
  rows: T[],
  dateValue: DateRangePickerValue | null | undefined,
): T[] {
  if (rows.length === 0) return [];
  if (isSingleDayRange(dateValue)) {
    return [collapseDailyResults(rows, SINGLE_DAY_TIME_LABEL)];
  }
  return [...rows].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
}
