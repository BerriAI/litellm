import type { DailyData } from "@/components/UsagePage/types";

export type CustomerUsageTotal = Readonly<{
  spend: number;
  requests: number;
  tokens: number;
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  cacheCost: number;
  pricedCacheTokens: number;
}>;

export type CustomerUsageTotals = ReadonlyMap<string, CustomerUsageTotal>;

export const EMPTY_CUSTOMER_USAGE: CustomerUsageTotal = {
  spend: 0,
  requests: 0,
  tokens: 0,
  inputTokens: 0,
  outputTokens: 0,
  cacheReadTokens: 0,
  cacheWriteTokens: 0,
  cacheCost: 0,
  pricedCacheTokens: 0,
};

export const customerModelBreakdown = ({ breakdown }: DailyData) =>
  Object.keys(breakdown.model_groups).length > 0 ? breakdown.model_groups : breakdown.models;

export function customerUsageTotals(days: readonly DailyData[]): CustomerUsageTotals {
  return days.reduce<CustomerUsageTotals>(
    (totals, day) =>
      new Map([
        ...totals,
        ...Object.entries(day.breakdown.entities).map(([id, entity]) => {
          const previous = totals.get(id) ?? EMPTY_CUSTOMER_USAGE;
          const metrics = entity.metrics;
          const next: CustomerUsageTotal = {
            spend: previous.spend + metrics.spend,
            requests: previous.requests + metrics.api_requests,
            tokens: previous.tokens + metrics.total_tokens,
            inputTokens: previous.inputTokens + metrics.prompt_tokens,
            outputTokens: previous.outputTokens + metrics.completion_tokens,
            cacheReadTokens: previous.cacheReadTokens + (metrics.cache_read_input_tokens ?? 0),
            cacheWriteTokens: previous.cacheWriteTokens + (metrics.cache_creation_input_tokens ?? 0),
            cacheCost: previous.cacheCost + (metrics.cache_cost ?? 0),
            pricedCacheTokens: previous.pricedCacheTokens + (metrics.priced_cache_tokens ?? 0),
          };
          return [id, next] as const;
        }),
      ]),
    new Map<string, CustomerUsageTotal>(),
  );
}
