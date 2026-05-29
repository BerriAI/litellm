import { DailyData, SpendMetrics, BreakdownMetrics, MetricWithMetadata, KeyMetricWithMetadata } from "../types";

/**
 * Numeric fields on `SpendMetrics` that should be summed when merging two
 * rows that share the same `date`.
 */
const SPEND_METRIC_KEYS: ReadonlyArray<keyof SpendMetrics> = [
  "spend",
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "api_requests",
  "successful_requests",
  "failed_requests",
  "cache_read_input_tokens",
  "cache_creation_input_tokens",
];

/** Top-level keys on `BreakdownMetrics` that hold `{ [name]: { metrics, ... } }` maps. */
const BREAKDOWN_MAP_KEYS: ReadonlyArray<keyof BreakdownMetrics> = [
  "models",
  "model_groups",
  "mcp_servers",
  "providers",
  "api_keys",
  "entities",
  "endpoints",
];

function emptyMetrics(): SpendMetrics {
  return {
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
}

function addMetrics(a: SpendMetrics | undefined, b: SpendMetrics | undefined): SpendMetrics {
  const out = emptyMetrics();
  for (const key of SPEND_METRIC_KEYS) {
    out[key] = (a?.[key] ?? 0) + (b?.[key] ?? 0);
  }
  return out;
}

function mergeApiKeyBreakdown(
  a: { [key: string]: KeyMetricWithMetadata } | undefined,
  b: { [key: string]: KeyMetricWithMetadata } | undefined,
): { [key: string]: KeyMetricWithMetadata } {
  const out: { [key: string]: KeyMetricWithMetadata } = {};
  const keys = new Set<string>([...Object.keys(a ?? {}), ...Object.keys(b ?? {})]);
  for (const k of keys) {
    const left = a?.[k];
    const right = b?.[k];
    out[k] = {
      metrics: addMetrics(left?.metrics, right?.metrics),
      // First-seen wins for metadata (alias/team_id/tags are static per-key).
      metadata: (left?.metadata ?? right?.metadata) as KeyMetricWithMetadata["metadata"],
    };
  }
  return out;
}

function mergeMetricWithMetadataMap<T extends MetricWithMetadata | KeyMetricWithMetadata>(
  a: { [key: string]: T } | undefined,
  b: { [key: string]: T } | undefined,
): { [key: string]: T } {
  const out: { [key: string]: T } = {};
  const keys = new Set<string>([...Object.keys(a ?? {}), ...Object.keys(b ?? {})]);
  for (const k of keys) {
    const left = a?.[k] as MetricWithMetadata | KeyMetricWithMetadata | undefined;
    const right = b?.[k] as MetricWithMetadata | KeyMetricWithMetadata | undefined;
    const mergedMetrics = addMetrics(left?.metrics, right?.metrics);
    // First-seen wins for metadata.
    const mergedMetadata = (left?.metadata ?? right?.metadata ?? {}) as any;

    const leftHasKeyBreakdown =
      left !== undefined && (left as MetricWithMetadata).api_key_breakdown !== undefined;
    const rightHasKeyBreakdown =
      right !== undefined && (right as MetricWithMetadata).api_key_breakdown !== undefined;

    if (leftHasKeyBreakdown || rightHasKeyBreakdown) {
      const merged: MetricWithMetadata = {
        metrics: mergedMetrics,
        metadata: mergedMetadata,
        api_key_breakdown: mergeApiKeyBreakdown(
          (left as MetricWithMetadata | undefined)?.api_key_breakdown,
          (right as MetricWithMetadata | undefined)?.api_key_breakdown,
        ),
      };
      out[k] = merged as T;
    } else {
      const merged: KeyMetricWithMetadata = {
        metrics: mergedMetrics,
        metadata: mergedMetadata,
      };
      out[k] = merged as T;
    }
  }
  return out;
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

function mergeBreakdowns(a: BreakdownMetrics, b: BreakdownMetrics): BreakdownMetrics {
  const out: BreakdownMetrics = emptyBreakdown();
  for (const k of BREAKDOWN_MAP_KEYS) {
    const leftMap = a?.[k];
    const rightMap = b?.[k];
    if (leftMap === undefined && rightMap === undefined) continue;
    const merged = mergeMetricWithMetadataMap(leftMap, rightMap);
    (out as any)[k] = merged;
  }
  return out;
}

function mergeTwo(a: DailyData, b: DailyData): DailyData {
  return {
    date: a.date,
    metrics: addMetrics(a.metrics, b.metrics),
    breakdown: mergeBreakdowns(a.breakdown ?? emptyBreakdown(), b.breakdown ?? emptyBreakdown()),
  };
}

/**
 * Collapse a list of `DailyData` rows so each unique `date` appears exactly
 * once, summing `metrics` and recursively merging `breakdown` maps.
 *
 * Required because `/user/daily/activity` is paginated and the same calendar
 * day can be split across pages — feeding the raw list into a date-indexed
 * `<BarChart />` produces one bar per row instead of one bar per date, which
 * makes single-day ranges (e.g. "Today") render N identical date labels.
 *
 * Insertion order is preserved (first occurrence of each date wins position).
 */
export function mergeDailyResults(results: DailyData[]): DailyData[] {
  if (!Array.isArray(results) || results.length === 0) return [];

  const order: string[] = [];
  const byDate = new Map<string, DailyData>();

  for (const row of results) {
    if (!row || typeof row.date !== "string") continue;
    const prior = byDate.get(row.date);
    if (prior === undefined) {
      order.push(row.date);
      // Clone so we never mutate the caller-supplied row.
      byDate.set(row.date, {
        date: row.date,
        metrics: addMetrics(undefined, row.metrics),
        breakdown: mergeBreakdowns(emptyBreakdown(), row.breakdown ?? emptyBreakdown()),
      });
    } else {
      byDate.set(row.date, mergeTwo(prior, row));
    }
  }

  return order.map((d) => byDate.get(d) as DailyData);
}
