import type { Team } from "@/components/key_team_helpers/key_list";
import i18n from "@/i18n/i18n";
import { processActivityData } from "@/components/activity_metrics";
import type { DailyActivityAggregatedResponse, DailyActivityMetadata, KeySpendActivityRow } from "./dailyActivityApi";
import { EMPTY_DAILY_ACTIVITY_METADATA, toDailyData } from "./dailyActivityApi";
import type { ModelActivityData, OtherModelData } from "./types";
import type { components } from "@/lib/http/schema";

export const reportingKeyId = (key: string): string => {
  if (!key) return "unattributed";
  return key === "unattributed" || key.startsWith("raw:") ? `raw:${key}` : key;
};

export const storedKeyId = (bucket: string): string => {
  if (bucket === "unattributed") return "";
  return bucket.startsWith("raw:") ? bucket.slice(4) : bucket;
};

export const overallUsageMetrics = (
  results: ReturnType<typeof toDailyData>,
  metadata: DailyActivityMetadata,
): ModelActivityData => ({
  label: i18n.t("usage.keyActivity.overallUsage"),
  total_requests: metadata.total_api_requests,
  total_successful_requests: metadata.total_successful_requests,
  total_failed_requests: metadata.total_failed_requests,
  total_cache_read_input_tokens: metadata.total_cache_read_input_tokens,
  total_cache_creation_input_tokens: metadata.total_cache_creation_input_tokens,
  total_tokens: metadata.total_tokens,
  prompt_tokens: metadata.total_prompt_tokens,
  completion_tokens: metadata.total_completion_tokens,
  total_spend: metadata.total_spend,
  total_response_time_ms: metadata.total_response_time_ms,
  total_timed_requests: metadata.total_timed_requests,
  top_models: [],
  daily_data: results
    .map((day) => ({
      date: day.date,
      metrics: {
        prompt_tokens: day.metrics.prompt_tokens,
        completion_tokens: day.metrics.completion_tokens,
        total_tokens: day.metrics.total_tokens,
        api_requests: day.metrics.api_requests,
        spend: day.metrics.spend,
        successful_requests: day.metrics.successful_requests,
        failed_requests: day.metrics.failed_requests,
        cache_read_input_tokens: day.metrics.cache_read_input_tokens,
        cache_creation_input_tokens: day.metrics.cache_creation_input_tokens,
        avg_response_time_ms:
          day.metrics.timed_requests && day.metrics.timed_requests > 0
            ? (day.metrics.total_response_time_ms ?? 0) / day.metrics.timed_requests
            : null,
      },
    }))
    .sort((a, b) => a.date.localeCompare(b.date)),
});

export const mergeKeyActivityPages = (
  current: readonly KeySpendActivityRow[],
  next: readonly KeySpendActivityRow[],
  total: number,
  offset: number,
): { rows: KeySpendActivityRow[]; nextOffset: number; hasMore: boolean } => {
  const rows: KeySpendActivityRow[] = Array.from(
    new Map([...current, ...next].map((row) => [row.api_key, row])).values(),
  );
  const nextOffset = offset + next.length;
  return { rows, nextOffset, hasMore: next.length > 0 && nextOffset < total };
};

export const keyDetailFromResponse = (
  response: DailyActivityAggregatedResponse,
  apiKey: string,
  teams: Team[],
): ModelActivityData | undefined => processActivityData({ results: toDailyData(response) }, "api_keys", teams)[apiKey];

const modelTotals = (metrics: components["schemas"]["SpendMetrics"]) => ({
  spend: metrics.spend ?? 0,
  requests: metrics.api_requests ?? 0,
  successful_requests: metrics.successful_requests ?? 0,
  failed_requests: metrics.failed_requests ?? 0,
  tokens: metrics.total_tokens ?? 0,
});

export const boundedActivityMetrics = (
  response: DailyActivityAggregatedResponse,
): { metrics: ModelActivityData; otherModels?: OtherModelData } => {
  const breakdown = "model_breakdown" in response ? response.model_breakdown : undefined;
  return {
    metrics: {
      ...overallUsageMetrics(toDailyData(response), response.metadata ?? EMPTY_DAILY_ACTIVITY_METADATA),
      top_models: breakdown?.models.map(({ model, metrics }) => ({ model, ...modelTotals(metrics) })) ?? [],
    },
    otherModels: breakdown?.other
      ? { ...modelTotals(breakdown.other.metrics), model_count: breakdown.other.model_count }
      : undefined,
  };
};
