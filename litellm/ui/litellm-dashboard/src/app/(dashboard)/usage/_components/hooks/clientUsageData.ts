import type { ParseKeys } from "i18next";
import { z } from "zod";
import {
  EMPTY_DAILY_ACTIVITY_METADATA,
  EMPTY_DAILY_ACTIVITY_RESPONSE,
  type DailyActivityAggregatedResponse,
} from "@/components/UsagePage/dailyActivityApi";
import i18n from "@/i18n/i18n";
import type { components } from "@/lib/http/schema";
import { extractPromptCacheTokens } from "@/utils/promptCacheUsage";
import { isUserAgentTag, usageClientFromUserAgent, UsageClient } from "../components/overview/agentCatalog";

type DailyData = components["schemas"]["DailySpendData"];
type MetricWithMetadata = components["schemas"]["MetricWithMetadata"];
type SpendMetrics = components["schemas"]["SpendMetrics"];

export { UsageClient } from "../components/overview/agentCatalog";
export const CLIENT_REQUEST_LIMIT = 1000;
export const INCOMPLETE_CLIENT_USAGE: ParseKeys = "usage.clientUsage.errors.narrowRange";

const count = z.number().int().nonnegative();
const cacheDetailsSchema = z.object({
  cached_tokens: count.nullish(),
  cache_creation_tokens: count.nullish(),
  cache_write_tokens: count.nullish(),
});
const usageFields = {
  cache_read_input_tokens: count.nullish(),
  cache_creation_input_tokens: count.nullish(),
  prompt_tokens_details: cacheDetailsSchema.nullish(),
  input_tokens_details: cacheDetailsSchema.nullish(),
};
const usageSchema = z.object(usageFields);
const guardrailEntrySchema = z.object({
  guardrail_cost: z.number().finite().nullish().catch(null),
  guardrail_cost_in_spend: z.boolean().nullish().catch(null),
});
const metadataFields = {
  user_agent: z.string().nullish(),
  status: z.string().nullish(),
  additional_usage_values: usageSchema.nullish(),
  usage_object: usageSchema.nullish(),
  guardrail_information: z
    .union([z.array(guardrailEntrySchema), guardrailEntrySchema])
    .nullish()
    .catch(null),
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
}

export const emptyClientUsage = (error: string | null = null): ClientUsageResult => ({
  data: EMPTY_DAILY_ACTIVITY_RESPONSE,
  error,
  cacheAvailable: false,
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
      if (isUserAgentTag(key)) return [key];
      return isUserAgentTag(`${key.trim()}:`) && typeof value === "string" ? [`User-Agent: ${value}`] : [];
    });
  }
  return [];
}

export function identifyUsageClient(metadataAgent: string | null | undefined, tags: unknown): UsageClient | null {
  const candidates = [...(metadataAgent ? [metadataAgent] : []), ...userAgentTags(tags).filter(isUserAgentTag)].map(
    usageClientFromUserAgent,
  );
  if (candidates.includes(UsageClient.ClaudeDesktop)) return UsageClient.ClaudeDesktop;
  if (candidates.includes(UsageClient.ClaudeCode)) return UsageClient.ClaudeCode;
  return null;
}

const statusOf = (row: ClientLog) => row.status ?? row.metadata?.status;
const cacheTokens = (row: ClientLog) => {
  const additional = extractPromptCacheTokens(row.metadata?.additional_usage_values, true);
  const usage = extractPromptCacheTokens(row.metadata?.usage_object, true);
  return {
    cacheReadTokens: additional.cacheReadTokens ?? usage.cacheReadTokens,
    cacheCreationTokens: additional.cacheCreationTokens ?? usage.cacheCreationTokens,
  };
};
const billedGuardrailCost = (entry: z.infer<typeof guardrailEntrySchema>) =>
  entry.guardrail_cost_in_spend === false ? 0 : Math.max(entry.guardrail_cost ?? 0, 0);
type GuardrailInformation = NonNullable<NonNullable<ClientLog["metadata"]>["guardrail_information"]>;
const guardrailEntries = (information: GuardrailInformation) =>
  Array.isArray(information) ? information : [information];
const guardrailSpend = (row: ClientLog) => {
  const information = row.metadata?.guardrail_information;
  const entries = information == null ? [] : guardrailEntries(information);
  return Math.min(
    entries.reduce((total, entry) => total + billedGuardrailCost(entry), 0),
    Math.max(row.spend, 0),
  );
};
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
  guardrail_spend: sum(rows, guardrailSpend),
  spend: sum(rows, (row) => row.spend),
  prompt_tokens: sum(rows, (row) => row.prompt_tokens),
  completion_tokens: sum(rows, (row) => row.completion_tokens),
  total_tokens: sum(rows, (row) => row.total_tokens),
  api_requests: rows.length,
  successful_requests: rows.filter((row) => statusOf(row) === "success").length,
  failed_requests: rows.filter((row) => statusOf(row) === "failure").length,
  cache_read_input_tokens: sum(rows, (row) => cacheTokens(row).cacheReadTokens ?? 0),
  cache_creation_input_tokens: sum(rows, (row) => cacheTokens(row).cacheCreationTokens ?? 0),
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
const modelName = (row: ClientLog) => row.model || i18n.t("usage.clientUsage.unknownModel");
const dailyData = (rows: readonly ClientLog[]): DailyData[] =>
  [...new Set(rows.map((row) => localDay(row.startTime)))].sort().map((date) => {
    const day = rows.filter((row) => localDay(row.startTime) === date);
    return {
      date,
      metrics: metricsFor(day),
      breakdown: {
        models: breakdownFor(day, modelName),
        model_groups: breakdownFor(day, (row) => row.model_group || modelName(row)),
        providers: breakdownFor(day, (row) => row.custom_llm_provider || i18n.t("usage.clientUsage.unknownProvider")),
        api_keys: {},
        mcp_servers: {},
        entities: {},
        endpoints: {},
      },
    };
  });

export function clientUsageFromLogs(response: unknown, scope: ClientUsageScope): ClientUsageResult {
  if (!scope.userId && !scope.teamId) return emptyClientUsage(i18n.t("usage.clientUsage.errors.selectUserOrTeam"));
  const parsed = pageSchema.safeParse(response);
  if (!parsed.success) return emptyClientUsage(i18n.t("usage.clientUsage.errors.incompleteData"));
  const page = parsed.data;
  if (page.total_is_capped || page.total_pages > 1 || page.total > CLIENT_REQUEST_LIMIT) {
    return emptyClientUsage(i18n.t(INCOMPLETE_CLIENT_USAGE));
  }
  if (page.data.length !== page.total) return emptyClientUsage(i18n.t("usage.clientUsage.errors.incompleteData"));
  if (scope.userId && page.data.some((row) => row.user !== scope.userId)) {
    return emptyClientUsage(i18n.t("usage.clientUsage.errors.outsideUser"));
  }
  if (scope.teamId && page.data.some((row) => row.team_id !== scope.teamId)) {
    return emptyClientUsage(i18n.t("usage.clientUsage.errors.outsideTeam"));
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
    cacheAvailable: rows.every((row) => {
      const cache = cacheTokens(row);
      return cache.cacheReadTokens !== undefined && cache.cacheCreationTokens !== undefined;
    }),
  };
}
