import type { DailyData } from "@/components/UsagePage/types";

export const CUSTOMER_TOKEN_COLUMNS = [
  { id: "tokens", label: "usage.statStrip.tokenPopover.totalTokens" },
  { id: "inputTokens", label: "usage.statStrip.tokenPopover.inputTokens" },
  { id: "outputTokens", label: "usage.statStrip.tokenPopover.outputTokens" },
  { id: "cacheReadTokens", label: "customerPortal.tokenUsage.cacheReadTokens" },
  { id: "cacheWriteTokens", label: "customerPortal.tokenUsage.cacheWriteTokens" },
] as const;

type Totals = Readonly<
  { spend: number; requests: number; cacheCost: number; pricedCacheTokens: number } & Record<
    (typeof CUSTOMER_TOKEN_COLUMNS)[number]["id"],
    number
  >
>;

export type CustomerUsageTotals = ReadonlyMap<string, Totals>;

const ZERO_TOTALS: Totals = {
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

export function customerUsageTotals(
  days: readonly DailyData[],
  selectBreakdown = (day: DailyData) => day.breakdown.entities,
): CustomerUsageTotals {
  return days.reduce((totals, day) => {
    for (const [id, entity] of Object.entries(selectBreakdown(day))) {
      const previous = totals.get(id) ?? ZERO_TOTALS;
      const metrics = entity.metrics;
      const next = {
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
      totals.set(id, next);
    }
    return totals;
  }, new Map<string, Totals>());
}

export function customerModelTokenUsage(days: readonly DailyData[]) {
  return [...customerUsageTotals(days, customerModelBreakdown)]
    .filter(([, totals]) => totals.tokens > 0)
    .map(([model, totals]) => ({
      model,
      ...totals,
      other:
        totals.cacheReadTokens + totals.cacheWriteTokens > Math.min(totals.tokens, totals.inputTokens)
          ? null
          : totals.tokens - totals.cacheReadTokens - totals.cacheWriteTokens,
    }))
    .sort((a, b) => b.tokens - a.tokens || a.model.localeCompare(b.model));
}
