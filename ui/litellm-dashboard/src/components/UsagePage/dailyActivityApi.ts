import type { components } from "@/lib/http/schema";
import type { BreakdownMetrics, DailyData, KeyMetadata, KeyMetricWithMetadata, MetricWithMetadata } from "./types";

export type DailyActivityEntity = "user" | "team" | "tag" | "organization" | "customer" | "agent";
export type DailyActivityAggregatedResponse = components["schemas"]["SpendAnalyticsPaginatedResponse"];
export type DailyActivityMetadata = components["schemas"]["DailySpendMetadata"];
export type ExportType = components["schemas"]["ExportType"];
export type ExportFormat = "csv" | "json";

export type KeyActivityRow = components["schemas"]["KeyActivityRow"];
export type KeySpendActivityRow = components["schemas"]["KeySpendActivityRow"];
export type DailyActivityKeySearchResponse = components["schemas"]["DailyActivityKeySearchResponse"];
export type DailyActivityKeyPageResponse = components["schemas"]["DailyActivityKeyPageResponse"];
export type ModelTopKeysResponse = components["schemas"]["ModelTopKeysResponse"];
export type CacheLeakageKeysResponse = components["schemas"]["CacheLeakageKeysResponse"];

export interface DailyActivityRequest {
  accessToken: string;
  startTime: Date;
  endTime: Date;
  entityIds?: readonly string[] | null;
  excludeEntityIds?: readonly string[];
  apiKey?: string | null;
  model?: string | null;
  includeCurrentUtcDay?: boolean;
  apiKeyLimit?: number;
}

export const EMPTY_DAILY_ACTIVITY_METADATA: DailyActivityMetadata = {
  has_more: false,
  page: 1,
  total_pages: 1,
  total_spend: 0,
  total_flat_cost: 0,
  total_api_requests: 0,
  total_successful_requests: 0,
  total_failed_requests: 0,
  total_tokens: 0,
  total_prompt_tokens: 0,
  total_completion_tokens: 0,
  total_cache_read_input_tokens: 0,
  total_cache_creation_input_tokens: 0,
  total_compression_saved_tokens: 0,
  total_compression_savings_spend: 0,
  total_prompt_caching_savings_spend: 0,
  total_gateway_injected_caching_savings_spend: 0,
  total_autorouter_savings_spend: 0,
  total_response_time_ms: 0,
  total_timed_requests: 0,
};

export const EMPTY_DAILY_ACTIVITY_RESPONSE: DailyActivityAggregatedResponse = {
  results: [],
  metadata: EMPTY_DAILY_ACTIVITY_METADATA,
};

type SchemaMetricWithMetadata = components["schemas"]["MetricWithMetadata"];
type SchemaKeyMetricWithMetadata = components["schemas"]["KeyMetricWithMetadata"];

const toKeyMetric = (entry: SchemaKeyMetricWithMetadata): KeyMetricWithMetadata => ({
  metrics: entry.metrics,
  metadata: toKeyMetadata(entry.metadata),
});

export const toKeyMetadata = (metadata: components["schemas"]["KeyMetadata"] | undefined): KeyMetadata => ({
  key_alias: metadata?.key_alias ?? null,
  team_id: metadata?.team_id ?? null,
  user_id: metadata?.user_id,
  user_email: metadata?.user_email,
  key_exists: metadata?.key_exists,
});

const toMetric = (entry: SchemaMetricWithMetadata): MetricWithMetadata => ({
  metrics: entry.metrics,
  metadata: entry.metadata ?? {},
  api_key_breakdown: Object.fromEntries(
    Object.entries(entry.api_key_breakdown ?? {}).map(([key, value]) => [key, toKeyMetric(value)]),
  ),
});

const toMetricMap = (
  map: { [key: string]: SchemaMetricWithMetadata } | undefined,
): { [key: string]: MetricWithMetadata } =>
  Object.fromEntries(Object.entries(map ?? {}).map(([key, value]) => [key, toMetric(value)]));

export const toDailyData = (response: DailyActivityAggregatedResponse): DailyData[] =>
  (response.results ?? []).map((day) => {
    const breakdown = day.breakdown;
    const normalized: BreakdownMetrics = {
      models: toMetricMap(breakdown?.models),
      model_groups: toMetricMap(breakdown?.model_groups),
      mcp_servers: toMetricMap(breakdown?.mcp_servers),
      providers: toMetricMap(breakdown?.providers),
      api_keys: Object.fromEntries(
        Object.entries(breakdown?.api_keys ?? {}).map(([key, value]) => [key, toKeyMetric(value)]),
      ),
      entities: toMetricMap(breakdown?.entities),
      endpoints: toMetricMap(breakdown?.endpoints),
    };
    return { date: day.date, metrics: day.metrics, breakdown: normalized };
  });
