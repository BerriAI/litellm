import { DailyData } from "@/components/UsagePage/types";
import { ToolSpendDailyEntry, ToolSpendEntry } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export const usd = (value: number): string => {
  const decimals = value > 0 && value < 1 ? 4 : 2;
  return `$${formatNumberWithCommas(value, decimals)}`;
};

export const pct = (ratio: number): string => `${formatNumberWithCommas(ratio * 100, 1)}%`;

export interface CacheLeakageRow {
  apiKey: string;
  keyAlias: string | null;
  teamId: string | null;
  uncachedPromptTokens: number;
  cacheReadTokens: number;
  cacheHitRatio: number;
  realizedCachingSavings: number;
  estSavingsLeft: number | null;
}

export interface CacheLeakageResult {
  rows: CacheLeakageRow[];
  discountPerToken: number | null;
}

interface KeyAccumulator {
  keyAlias: string | null;
  teamId: string | null;
  promptTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  realizedCachingSavings: number;
}

const emptyAccumulator = (): KeyAccumulator => ({
  keyAlias: null,
  teamId: null,
  promptTokens: 0,
  cacheReadTokens: 0,
  cacheCreationTokens: 0,
  realizedCachingSavings: 0,
});

export const computeCacheLeakage = (results: readonly DailyData[], limit = 10): CacheLeakageResult => {
  const byKey = new Map<string, KeyAccumulator>();
  for (const day of results) {
    const apiKeys = day.breakdown?.api_keys ?? {};
    for (const [apiKey, entry] of Object.entries(apiKeys)) {
      const acc = byKey.get(apiKey) ?? emptyAccumulator();
      const m = entry.metrics;
      const next: KeyAccumulator = {
        keyAlias: acc.keyAlias ?? entry.metadata?.key_alias ?? null,
        teamId: acc.teamId ?? entry.metadata?.team_id ?? null,
        promptTokens: acc.promptTokens + (m.prompt_tokens ?? 0),
        cacheReadTokens: acc.cacheReadTokens + (m.cache_read_input_tokens ?? 0),
        cacheCreationTokens: acc.cacheCreationTokens + (m.cache_creation_input_tokens ?? 0),
        realizedCachingSavings: acc.realizedCachingSavings + (m.prompt_caching_savings_spend ?? 0),
      };
      byKey.set(apiKey, next);
    }
  }

  const totals = [...byKey.values()].reduce(
    (agg, a) => ({
      cacheReadTokens: agg.cacheReadTokens + a.cacheReadTokens,
      realizedCachingSavings: agg.realizedCachingSavings + a.realizedCachingSavings,
    }),
    { cacheReadTokens: 0, realizedCachingSavings: 0 },
  );
  const discountPerToken = totals.cacheReadTokens > 0 ? totals.realizedCachingSavings / totals.cacheReadTokens : null;

  const rows: CacheLeakageRow[] = [...byKey.entries()]
    .map(([apiKey, a]) => {
      const uncachedPromptTokens = Math.max(0, a.promptTokens - a.cacheReadTokens - a.cacheCreationTokens);
      return {
        apiKey,
        keyAlias: a.keyAlias,
        teamId: a.teamId,
        uncachedPromptTokens,
        cacheReadTokens: a.cacheReadTokens,
        cacheHitRatio: a.promptTokens > 0 ? a.cacheReadTokens / a.promptTokens : 0,
        realizedCachingSavings: a.realizedCachingSavings,
        estSavingsLeft: discountPerToken != null ? uncachedPromptTokens * discountPerToken : null,
      };
    })
    .filter((row) => row.uncachedPromptTokens > 0);

  const sorted = rows.sort((x, y) =>
    discountPerToken != null
      ? (y.estSavingsLeft ?? 0) - (x.estSavingsLeft ?? 0)
      : y.uncachedPromptTokens - x.uncachedPromptTokens,
  );

  return { rows: sorted.slice(0, limit), discountPerToken };
};

export interface DailyToolSpendPoint {
  date: string;
  [toolName: string]: string | number;
}

export const buildDailyToolSeries = (
  daily: readonly ToolSpendDailyEntry[],
  topToolNames: readonly string[],
): DailyToolSpendPoint[] => {
  const top = new Set(topToolNames);
  const byDate = new Map<string, DailyToolSpendPoint>();
  for (const d of daily) {
    if (!top.has(d.tool_name)) continue;
    const point = byDate.get(d.date) ?? seedPoint(d.date, topToolNames);
    point[d.tool_name] = (Number(point[d.tool_name]) || 0) + d.spend;
    byDate.set(d.date, point);
  }
  return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
};

const seedPoint = (date: string, toolNames: readonly string[]): DailyToolSpendPoint =>
  toolNames.reduce<DailyToolSpendPoint>((p, name) => ({ ...p, [name]: 0 }), { date });

export const topToolsBySpend = (byTool: readonly ToolSpendEntry[], limit = 8): ToolSpendEntry[] =>
  [...byTool].sort((a, b) => b.spend - a.spend).slice(0, limit);
