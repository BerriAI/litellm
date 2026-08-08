import type {
  BreakdownMetrics,
  DailyData,
  KeyMetricWithMetadata,
  MetricWithMetadata,
  SpendMetrics,
} from "@/components/UsagePage/types";

const METRIC_KEYS: readonly (keyof SpendMetrics)[] = [
  "spend",
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "api_requests",
  "successful_requests",
  "failed_requests",
  "cache_read_input_tokens",
  "cache_creation_input_tokens",
  "compression_saved_tokens",
  "compression_savings_spend",
  "prompt_caching_savings_spend",
  "autorouter_savings_spend",
];

const addMetrics = (a: SpendMetrics, b: SpendMetrics): SpendMetrics =>
  METRIC_KEYS.reduce(
    (acc, key) =>
      a[key] === undefined && b[key] === undefined ? acc : { ...acc, [key]: (a[key] ?? 0) + (b[key] ?? 0) },
    {} as SpendMetrics,
  );

const mergeBuckets = <T>(
  a: Record<string, T> | undefined,
  b: Record<string, T> | undefined,
  mergeEntry: (left: T, right: T) => T,
): Record<string, T> => {
  const left = a ?? {};
  const right = b ?? {};
  return Object.fromEntries(
    Array.from(new Set([...Object.keys(left), ...Object.keys(right)])).map((key) => {
      const leftEntry = left[key];
      const rightEntry = right[key];
      if (leftEntry === undefined) return [key, rightEntry];
      if (rightEntry === undefined) return [key, leftEntry];
      return [key, mergeEntry(leftEntry, rightEntry)];
    }),
  );
};

const mergeKeyMetric = (a: KeyMetricWithMetadata, b: KeyMetricWithMetadata): KeyMetricWithMetadata => ({
  ...a,
  metrics: addMetrics(a.metrics, b.metrics),
});

const mergeMetricWithMetadata = (a: MetricWithMetadata, b: MetricWithMetadata): MetricWithMetadata => ({
  ...a,
  metrics: addMetrics(a.metrics, b.metrics),
  api_key_breakdown: mergeBuckets(a.api_key_breakdown, b.api_key_breakdown, mergeKeyMetric),
});

const mergeBreakdown = (a: BreakdownMetrics, b: BreakdownMetrics): BreakdownMetrics => ({
  models: mergeBuckets(a.models, b.models, mergeMetricWithMetadata),
  model_groups: mergeBuckets(a.model_groups, b.model_groups, mergeMetricWithMetadata),
  mcp_servers: mergeBuckets(a.mcp_servers, b.mcp_servers, mergeMetricWithMetadata),
  providers: mergeBuckets(a.providers, b.providers, mergeMetricWithMetadata),
  entities: mergeBuckets(a.entities, b.entities, mergeMetricWithMetadata),
  endpoints: mergeBuckets(a.endpoints, b.endpoints, mergeMetricWithMetadata),
  api_keys: mergeBuckets(a.api_keys, b.api_keys, mergeKeyMetric),
});

const mergeDay = (a: DailyData, b: DailyData): DailyData => ({
  ...a,
  metrics: addMetrics(a.metrics, b.metrics),
  breakdown: mergeBreakdown(a.breakdown, b.breakdown),
});

/**
 * Combine daily activity pages into one series with a single entry per date.
 *
 * The backend paginates over raw spend rows, so a date whose rows straddle a
 * page boundary comes back once per page, each entry holding only that page's
 * share of the day. Concatenating those entries leaves duplicate dates that
 * under-report every per-day figure in the charts and the CSV export.
 */
export const mergeDailyResults = (existing: readonly DailyData[], incoming: readonly DailyData[]): DailyData[] =>
  incoming.reduce<DailyData[]>(
    (acc, day) => {
      const index = acc.findIndex((existingDay) => existingDay.date === day.date);
      if (index === -1) return [...acc, day];
      return acc.map((existingDay, i) => (i === index ? mergeDay(existingDay, day) : existingDay));
    },
    [...existing],
  );
