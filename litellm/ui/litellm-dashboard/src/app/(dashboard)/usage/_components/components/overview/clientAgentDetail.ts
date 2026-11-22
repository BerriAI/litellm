import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { AgentRow } from "./agentCatalog";
import { reconciledTokenBreakdown, type TokenBreakdown } from "./tokenBreakdown";

export interface ClientAgentDetail {
  summary: Pick<AgentRow, "spend" | "tokens" | "requests">;
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
  const metrics = days.flatMap((day) =>
    tags.flatMap((tag) => {
      const entry = day.breakdown.entities[tag];
      return entry ? [entry.metrics] : [];
    }),
  );
  const spend = knownTotal(metrics, "spend");
  const spendMatches = spend !== null && Math.abs(spend - agent.spend) <= Math.max(1e-8, Math.abs(agent.spend) * 1e-9);
  const countsMatch =
    knownTotal(metrics, "total_tokens") === agent.tokens && knownTotal(metrics, "api_requests") === agent.requests;
  const dailyMatchesSummary = spendMatches && countsMatch;
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
    dailyMatchesSummary,
    tokenBreakdown: reconciledTokenBreakdown(agent.tokens, tokenBreakdown),
  };
};
