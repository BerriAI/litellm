import { z } from "zod";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  EMPTY_DAILY_ACTIVITY_RESPONSE,
  type DailyActivityAggregatedResponse,
} from "@/components/UsagePage/dailyActivityApi";
import type { components } from "@/lib/http/schema";

type DailyData = components["schemas"]["DailySpendData"];
type MetricWithMetadata = components["schemas"]["MetricWithMetadata"];
type SpendMetrics = components["schemas"]["SpendMetrics"];

export type UsageClient = "claude-desktop" | "claude-code";
export const CLIENT_REQUEST_LIMIT = 1000;
export const INCOMPLETE_CLIENT_USAGE =
  "Narrow the date range to fewer than 1,000 requests to see complete client usage";

const count = z.number().int().nonnegative();
const usageSchema = z.object({
  cache_read_input_tokens: count.nullish(),
  cache_creation_input_tokens: count.nullish(),
  prompt_tokens_details: z
    .object({
      cached_tokens: count.nullish(),
      cache_creation_tokens: count.nullish(),
      cache_write_tokens: count.nullish(),
    })
    .nullish(),
});
const metadataFields = {
  user_agent: z.string().nullish(),
  status: z.string().nullish(),
  additional_usage_values: usageSchema.nullish(),
  usage_object: usageSchema.nullish(),
};
const logFields = {
  request_id: z.string().min(1),
  user: z.string().nullish(),
  team_id: z.string().nullish(),
  startTime: z
    .string()
    .transform((value) => {
      const withZone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? value : `${value}Z`;
      return new Date(withZone);
    })
    .refine((value) => Number.isFinite(value.getTime())),
  model: z.string().nullish(),
  model_group: z.string().nullish(),
  custom_llm_provider: z.string().nullish(),
  spend: z.number().finite(),
  total_tokens: count,
  prompt_tokens: count,
  completion_tokens: count,
  request_duration_ms: z.number().nonnegative().nullish(),
  status: z.string().nullish(),
  request_tags: z.unknown(),
  metadata: z.object(metadataFields).nullish(),
};
const logSchema = z.object(logFields);
const pageFields = {
  data: z.array(logSchema),
  total: count,
  page: z.literal(1),
  total_pages: count,
  total_is_capped: z.boolean(),
};
const pageSchema = z.object(pageFields);
type ClientLog = z.infer<typeof logSchema>;

export interface ClientUsageScope {
  userId: string | null;
  teamId?: string | null;
  client: UsageClient;
  startTime: Date;
  endTime: Date;
}

export interface ClientUsageResult {
  data: DailyActivityAggregatedResponse;
  error: string | null;
  cacheAvailable: boolean;
  statusAvailable: boolean;
}

export const emptyClientUsage = (error: string | null = null): ClientUsageResult => ({
  data: EMPTY_DAILY_ACTIVITY_RESPONSE,
  error,
  cacheAvailable: false,
  statusAvailable: false,
});

function userAgentTags(tags: unknown): string[] {
  if (Array.isArray(tags)) return tags.filter((tag): tag is string => typeof tag === "string");
  if (typeof tags === "string") {
    try {
      return userAgentTags(JSON.parse(tags));
    } catch {
      return [tags];
    }
  }
  if (tags !== null && typeof tags === "object") {
    return Object.entries(tags).flatMap(([key, value]) => {
      if (/^user-agent$/i.test(key) && typeof value === "string") return [`User-Agent: ${value}`];
      return /^user-agent:/i.test(key) ? [key] : [];
    });
  }
  return [];
}

export function identifyUsageClient(metadataAgent: string | null | undefined, tags: unknown): UsageClient | null {
  const candidates = [
    ...(metadataAgent ? [metadataAgent] : []),
    ...userAgentTags(tags)
      .filter((tag) => /^user-agent:/i.test(tag))
      .map((tag) => tag.replace(/^user-agent:\s*/i, "")),
  ];
  if (candidates.some((value) => /^claude[-_ ]desktop\b/i.test(value) || /\bclaude-desktop-3p\b/i.test(value))) {
    return "claude-desktop";
  }
  if (candidates.some((value) => /^(claude-code|claude_code|claude-user)\b/i.test(value))) return "claude-code";
  if (candidates.some((value) => /^claude-cli\//i.test(value) && /(?:\(|,)\s*cli\s*(?:,|\))/i.test(value))) {
    return "claude-code";
  }
  return null;
}

const statusOf = (row: ClientLog) => row.status ?? row.metadata?.status;
const recordedCacheTokens = (
  usage: z.infer<typeof usageSchema> | null | undefined,
  kind: "read" | "creation",
): number | null => {
  if (!usage) return null;
  const details = usage.prompt_tokens_details;
  if (kind === "read") return usage.cache_read_input_tokens ?? details?.cached_tokens ?? null;
  return usage.cache_creation_input_tokens ?? details?.cache_creation_tokens ?? details?.cache_write_tokens ?? null;
};
const cacheTokens = (row: ClientLog, kind: "read" | "creation"): number | null =>
  recordedCacheTokens(row.metadata?.additional_usage_values, kind) ??
  recordedCacheTokens(row.metadata?.usage_object, kind);
const sum = (rows: readonly ClientLog[], value: (row: ClientLog) => number) =>
  rows.reduce((total, row) => total + value(row), 0);
const metricsFor = (rows: readonly ClientLog[]): SpendMetrics => ({
  flat_cost: 0,
  ptu_hours: 0,
  compression_saved_tokens: 0,
  compression_savings_spend: 0,
  prompt_caching_savings_spend: 0,
  gateway_injected_caching_savings_spend: 0,
  autorouter_savings_spend: 0,
  spend: sum(rows, (row) => row.spend),
  prompt_tokens: sum(rows, (row) => row.prompt_tokens),
  completion_tokens: sum(rows, (row) => row.completion_tokens),
  total_tokens: sum(rows, (row) => row.total_tokens),
  api_requests: rows.length,
  successful_requests: rows.filter((row) => statusOf(row) === "success").length,
  failed_requests: rows.filter((row) => statusOf(row) === "failure").length,
  cache_read_input_tokens: sum(rows, (row) => cacheTokens(row, "read") ?? 0),
  cache_creation_input_tokens: sum(rows, (row) => cacheTokens(row, "creation") ?? 0),
  total_response_time_ms: sum(rows, (row) => (statusOf(row) === "success" ? row.request_duration_ms ?? 0 : 0)),
  timed_requests: rows.filter((row) => statusOf(row) === "success" && row.request_duration_ms != null).length,
});
const breakdownFor = (
  rows: readonly ClientLog[],
  dimension: (row: ClientLog) => string,
): Record<string, MetricWithMetadata> =>
  Object.fromEntries(
    [...new Set(rows.map(dimension))].map((name) => [
      name,
      {
        metrics: metricsFor(rows.filter((row) => dimension(row) === name)),
        metadata: {},
        api_key_breakdown: {},
      },
    ]),
  );

const localDay = (date: Date) =>
  `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
const modelName = (row: ClientLog) => row.model || "Unknown model";
const dailyData = (rows: readonly ClientLog[]): DailyData[] =>
  [...new Set(rows.map((row) => localDay(row.startTime)))].sort().map((date) => {
    const day = rows.filter((row) => localDay(row.startTime) === date);
    return {
      date,
      metrics: metricsFor(day),
      breakdown: {
        models: breakdownFor(day, modelName),
        model_groups: breakdownFor(day, (row) => row.model_group || modelName(row)),
        providers: breakdownFor(day, (row) => row.custom_llm_provider || "Unknown provider"),
        api_keys: {},
        mcp_servers: {},
        entities: {},
        endpoints: {},
      },
    };
  });

export function clientUsageFromLogs(response: unknown, scope: ClientUsageScope): ClientUsageResult {
  if (!scope.userId && !scope.teamId) return emptyClientUsage("Select a user or team to view client usage");
  const parsed = pageSchema.safeParse(response);
  if (!parsed.success) return emptyClientUsage("The proxy returned incomplete client usage data");
  const page = parsed.data;
  if (page.total_is_capped || page.total_pages > 1 || page.total > CLIENT_REQUEST_LIMIT) {
    return emptyClientUsage(INCOMPLETE_CLIENT_USAGE);
  }
  if (page.data.length !== page.total) return emptyClientUsage("The proxy returned incomplete client usage data");
  if (scope.userId && page.data.some((row) => row.user !== scope.userId)) {
    return emptyClientUsage("The proxy returned usage outside the selected user");
  }
  if (scope.teamId && page.data.some((row) => row.team_id !== scope.teamId)) {
    return emptyClientUsage("The proxy returned usage outside the selected team");
  }
  const uniqueRows = [...new Map(page.data.map((row) => [row.request_id, row])).values()];
  const rows = uniqueRows.filter(
    (row) =>
      row.startTime >= scope.startTime &&
      row.startTime <= scope.endTime &&
      identifyUsageClient(row.metadata?.user_agent, row.request_tags) === scope.client,
  );
  const metrics = metricsFor(rows);
  return {
    data: {
      results: dailyData(rows),
      metadata: {
        ...EMPTY_DAILY_ACTIVITY_METADATA,
        total_spend: metrics.spend,
        total_tokens: metrics.total_tokens,
        total_prompt_tokens: metrics.prompt_tokens,
        total_completion_tokens: metrics.completion_tokens,
        total_api_requests: metrics.api_requests,
        total_successful_requests: metrics.successful_requests,
        total_failed_requests: metrics.failed_requests,
        total_cache_read_input_tokens: metrics.cache_read_input_tokens,
        total_cache_creation_input_tokens: metrics.cache_creation_input_tokens,
        total_response_time_ms: metrics.total_response_time_ms,
        total_timed_requests: metrics.timed_requests,
      },
    },
    error: null,
    cacheAvailable: rows.every((row) => cacheTokens(row, "read") !== null && cacheTokens(row, "creation") !== null),
    statusAvailable: rows.every((row) => statusOf(row) === "success" || statusOf(row) === "failure"),
  };
}
