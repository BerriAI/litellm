import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { AgentRow } from "./agentCatalog";
import { reconciledTokenBreakdown, type TokenBreakdown } from "./tokenBreakdown";

interface ClientAgentTotals {
  spend: number;
  tokens: number;
  requests: number;
}

export interface ClientAgentDetail {
  summary: ClientAgentTotals;
  daily: (ClientAgentTotals & { date: string })[];
  dailyAvailable: boolean;
  dailyMatchesSummary: boolean;
  tokenBreakdown: TokenBreakdown;
}

const knownTotal = (metrics: readonly SpendMetrics[], key: keyof SpendMetrics): number | null => {
  const values = metrics.map((entry) => entry[key]);
  return values.length > 0 && values.every((value) => typeof value === "number" && Number.isFinite(value))
    ? values.reduce<number>((sum, value) => sum + (value ?? 0), 0)
    : null;
};

export const clientAgentDetail = (agent: AgentRow, days: readonly DailyData[]): ClientAgentDetail => {
  const tags = [...new Set(agent.tags)];
  const selected = days.map((day) => ({
    date: day.date,
    metrics: tags.flatMap((tag) => {
      const entry = day.breakdown.entities[tag];
      return entry ? [entry.metrics] : [];
    }),
  }));
  const metrics = selected.flatMap((day) => day.metrics);
  const daily = selected
    .map((day) => ({
      date: day.date,
      spend: day.metrics.reduce((sum, entry) => sum + entry.spend, 0),
      tokens: day.metrics.reduce((sum, entry) => sum + entry.total_tokens, 0),
      requests: day.metrics.reduce((sum, entry) => sum + entry.api_requests, 0),
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
  const dailyAvailable =
    metrics.length > 0 && daily.every((day) => [day.spend, day.tokens, day.requests].every(Number.isFinite));
  const spendMatches =
    Math.abs((knownTotal(metrics, "spend") ?? 0) - agent.spend) <= Math.max(1e-8, Math.abs(agent.spend) * 1e-9);
  const countsMatch =
    knownTotal(metrics, "total_tokens") === agent.tokens && knownTotal(metrics, "api_requests") === agent.requests;
  const dailyMatchesSummary = dailyAvailable && spendMatches && countsMatch;
  const tokenTotal = (key: keyof SpendMetrics) => {
    const total = dailyMatchesSummary ? knownTotal(metrics, key) : null;
    return total !== null && total >= 0 ? total : null;
  };
  const inputTokens = tokenTotal("prompt_tokens");
  const outputTokens = tokenTotal("completion_tokens");
  const cacheReadTokens = tokenTotal("cache_read_input_tokens");
  const cacheWriteTokens = tokenTotal("cache_creation_input_tokens");
  const tokenBreakdown = { inputTokens, outputTokens, cacheReadTokens, cacheWriteTokens };

  return {
    summary: { spend: agent.spend, tokens: agent.tokens, requests: agent.requests },
    daily: dailyAvailable ? daily : [],
    dailyAvailable,
    dailyMatchesSummary,
    tokenBreakdown: reconciledTokenBreakdown(agent.tokens, tokenBreakdown),
  };
};
