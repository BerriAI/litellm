import {
  BreakdownMetrics,
  DailyData,
  KeyMetricWithMetadata,
  MetricWithMetadata,
  SpendMetrics,
} from "../types";

/**
 * Keys of `SpendMetrics` that are numeric and additive across rows sharing the
 * same `date`. Tuple is constrained to `keyof SpendMetrics` so TypeScript
 * catches drift if `SpendMetrics` ever changes shape.
 */
const SUMMABLE_METRIC_KEYS = [
  "spend",
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "api_requests",
  "successful_requests",
  "failed_requests",
  "cache_read_input_tokens",
  "cache_creation_input_tokens",
] as const satisfies readonly (keyof SpendMetrics)[];

const EMPTY_METRICS = (): SpendMetrics => ({
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

function addMetrics(a: SpendMetrics, b: SpendMetrics): SpendMetrics {
  const out = EMPTY_METRICS();
  for (const k of SUMMABLE_METRIC_KEYS) {
    out[k] = (a[k] ?? 0) + (b[k] ?? 0);
  }
  return out;
}

function mergeKeyMetricWithMetadata(
  a: KeyMetricWithMetadata | undefined,
  b: KeyMetricWithMetadata,
): KeyMetricWithMetadata {
  if (!a) {
    return {
      metrics: { ...b.metrics },
      metadata: {
        key_alias: b.metadata?.key_alias ?? null,
        team_id: b.metadata?.team_id ?? null,
        tags: b.metadata?.tags ? [...b.metadata.tags] : undefined,
      },
    };
  }
  return {
    metrics: addMetrics(a.metrics, b.metrics),
    metadata: {
      key_alias: a.metadata.key_alias ?? b.metadata?.key_alias ?? null,
      team_id: a.metadata.team_id ?? b.metadata?.team_id ?? null,
      tags: b.metadata?.tags ?? a.metadata.tags,
    },
  };
}

function mergeApiKeyBreakdown(
  a: { [key: string]: KeyMetricWithMetadata },
  b: { [key: string]: KeyMetricWithMetadata },
): { [key: string]: KeyMetricWithMetadata } {
  const out: { [key: string]: KeyMetricWithMetadata } = {};
  for (const [k, v] of Object.entries(a)) out[k] = mergeKeyMetricWithMetadata(undefined, v);
  for (const [k, v] of Object.entries(b)) out[k] = mergeKeyMetricWithMetadata(out[k], v);
  return out;
}

function mergeMetricWithMetadata(
  a: MetricWithMetadata | undefined,
  b: MetricWithMetadata,
): MetricWithMetadata {
  if (!a) {
    return {
      metrics: { ...b.metrics },
      metadata: { ...((b.metadata as object) ?? {}) },
      api_key_breakdown: mergeApiKeyBreakdown({}, b.api_key_breakdown || {}),
    };
  }
  return {
    metrics: addMetrics(a.metrics, b.metrics),
    // Later row wins on metadata keys it explicitly provides — matches the
    // existing per-page `sumMetadata` behaviour in `usePaginatedDailyActivity`.
    metadata: { ...((a.metadata as object) ?? {}), ...((b.metadata as object) ?? {}) },
    api_key_breakdown: mergeApiKeyBreakdown(a.api_key_breakdown || {}, b.api_key_breakdown || {}),
  };
}

function mergeMetricMap(
  a: { [key: string]: MetricWithMetadata } | undefined,
  b: { [key: string]: MetricWithMetadata } | undefined,
): { [key: string]: MetricWithMetadata } {
  const out: { [key: string]: MetricWithMetadata } = {};
  for (const [k, v] of Object.entries(a || {})) out[k] = mergeMetricWithMetadata(undefined, v);
  for (const [k, v] of Object.entries(b || {})) out[k] = mergeMetricWithMetadata(out[k], v);
  return out;
}

function mergeKeyMetricMap(
  a: { [key: string]: KeyMetricWithMetadata } | undefined,
  b: { [key: string]: KeyMetricWithMetadata } | undefined,
): { [key: string]: KeyMetricWithMetadata } {
  const out: { [key: string]: KeyMetricWithMetadata } = {};
  for (const [k, v] of Object.entries(a || {})) out[k] = mergeKeyMetricWithMetadata(undefined, v);
  for (const [k, v] of Object.entries(b || {})) out[k] = mergeKeyMetricWithMetadata(out[k], v);
  return out;
}

function mergeBreakdown(a: BreakdownMetrics, b: BreakdownMetrics): BreakdownMetrics {
  const merged: BreakdownMetrics = {
    models: mergeMetricMap(a.models, b.models),
    model_groups: mergeMetricMap(a.model_groups, b.model_groups),
    mcp_servers: mergeMetricMap(a.mcp_servers, b.mcp_servers),
    providers: mergeMetricMap(a.providers, b.providers),
    api_keys: mergeKeyMetricMap(a.api_keys, b.api_keys),
    entities: mergeMetricMap(a.entities, b.entities),
  };
  if (a.endpoints || b.endpoints) {
    merged.endpoints = mergeMetricMap(a.endpoints, b.endpoints);
  }
  return merged;
}

const EMPTY_BREAKDOWN = (): BreakdownMetrics => ({
  models: {},
  model_groups: {},
  mcp_servers: {},
  providers: {},
  api_keys: {},
  entities: {},
});

/**
 * Collapse multiple `DailyData` rows that share the same `date` into a single
 * row.
 *
 * Why: the Daily Spend bar chart uses `date` as the BarChart `index`. When the
 * paginated `/user/daily/activity` accumulator concatenates rows across pages
 * (`usePaginatedDailyActivity.ts`), pages that all fall on the same date —
 * which is exactly what happens for the `Today` range — produce one bar per
 * page, all wearing the same X-axis label. The backend timezone window can
 * also emit a ±1 day ghost-row for the same reason.
 *
 * This function sums `metrics` and recursively merges every `breakdown.*` map
 * so downstream aggregations (top models, top keys, provider spend, etc.) see
 * exactly one row per date. First-seen date order is preserved.
 *
 * Pure & deterministic; no-op on already-deduped input.
 */
export function mergeResultsByDate(results: DailyData[]): DailyData[] {
  if (results.length < 2) return results.slice();
  const byDate = new Map<string, DailyData>();
  for (const row of results) {
    const existing = byDate.get(row.date);
    if (!existing) {
      byDate.set(row.date, {
        date: row.date,
        metrics: { ...row.metrics },
        breakdown: mergeBreakdown(EMPTY_BREAKDOWN(), row.breakdown),
      });
      continue;
    }
    byDate.set(row.date, {
      date: existing.date,
      metrics: addMetrics(existing.metrics, row.metrics),
      breakdown: mergeBreakdown(existing.breakdown, row.breakdown),
    });
  }
  return Array.from(byDate.values());
}
