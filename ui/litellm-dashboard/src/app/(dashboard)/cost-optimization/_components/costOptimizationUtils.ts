import { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import { ToolSpendDailyEntry, ToolSpendEntry } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export const usd = (value: number): string => {
  const decimals = value > 0 && value < 1 ? 4 : 2;
  return `$${formatNumberWithCommas(value, decimals)}`;
};

export const pct = (ratio: number): string => `${formatNumberWithCommas(ratio * 100, 1)}%`;

export type CacheLeakageDimension = "key" | "model";

export interface CacheLeakageRow {
  id: string;
  label: string;
  sublabel: string | null;
  uncachedPromptTokens: number;
  cacheHitRatio: number;
  potentialSavings: number | null;
}

export interface CacheLeakageResult {
  rows: CacheLeakageRow[];
  discountPerToken: number | null;
}

export const isAnthropicModel = (model: string): boolean => /claude|anthropic/i.test(model);

interface LeakageAccumulator {
  alias: string | null;
  teamId: string | null;
  promptTokens: number;
  cacheReadTokens: number;
  cacheCreationTokens: number;
  realizedCachingSavings: number;
}

const emptyAccumulator = (): LeakageAccumulator => ({
  alias: null,
  teamId: null,
  promptTokens: 0,
  cacheReadTokens: 0,
  cacheCreationTokens: 0,
  realizedCachingSavings: 0,
});

const addMetrics = (
  acc: LeakageAccumulator,
  m: SpendMetrics,
  alias: string | null,
  teamId: string | null,
): LeakageAccumulator => ({
  alias: acc.alias ?? alias,
  teamId: acc.teamId ?? teamId,
  promptTokens: acc.promptTokens + (m.prompt_tokens ?? 0),
  cacheReadTokens: acc.cacheReadTokens + (m.cache_read_input_tokens ?? 0),
  cacheCreationTokens: acc.cacheCreationTokens + (m.cache_creation_input_tokens ?? 0),
  realizedCachingSavings: acc.realizedCachingSavings + (m.prompt_caching_savings_spend ?? 0),
});

const aggregateByKey = (results: readonly DailyData[]): Map<string, LeakageAccumulator> => {
  const byKey = new Map<string, LeakageAccumulator>();
  for (const day of results) {
    for (const [apiKey, entry] of Object.entries(day.breakdown?.api_keys ?? {})) {
      const acc = byKey.get(apiKey) ?? emptyAccumulator();
      byKey.set(
        apiKey,
        addMetrics(acc, entry.metrics, entry.metadata?.key_alias ?? null, entry.metadata?.team_id ?? null),
      );
    }
  }
  return byKey;
};

const aggregateByModel = (results: readonly DailyData[]): Map<string, LeakageAccumulator> => {
  const byModel = new Map<string, LeakageAccumulator>();
  for (const day of results) {
    for (const [model, entry] of Object.entries(day.breakdown?.models ?? {})) {
      if (!isAnthropicModel(model)) continue;
      const acc = byModel.get(model) ?? emptyAccumulator();
      byModel.set(model, addMetrics(acc, entry.metrics, null, null));
    }
  }
  return byModel;
};

export const computeCacheLeakage = (
  results: readonly DailyData[],
  dimension: CacheLeakageDimension = "key",
  limit = 10,
): CacheLeakageResult => {
  const byEntity = dimension === "model" ? aggregateByModel(results) : aggregateByKey(results);

  const totals = [...byEntity.values()].reduce(
    (agg, a) => ({
      cacheReadTokens: agg.cacheReadTokens + a.cacheReadTokens,
      realizedCachingSavings: agg.realizedCachingSavings + a.realizedCachingSavings,
    }),
    { cacheReadTokens: 0, realizedCachingSavings: 0 },
  );
  const discountPerToken = totals.cacheReadTokens > 0 ? totals.realizedCachingSavings / totals.cacheReadTokens : null;

  const rows: CacheLeakageRow[] = [...byEntity.entries()]
    .map(([id, a]) => {
      const uncachedPromptTokens = Math.max(0, a.promptTokens - a.cacheReadTokens - a.cacheCreationTokens);
      return {
        id,
        label: dimension === "model" ? id : a.alias ?? `${id.slice(0, 8)}...`,
        sublabel: dimension === "model" ? null : a.teamId,
        uncachedPromptTokens,
        cacheHitRatio: a.promptTokens > 0 ? a.cacheReadTokens / a.promptTokens : 0,
        potentialSavings: discountPerToken != null ? uncachedPromptTokens * discountPerToken : null,
      };
    })
    .filter((row) => row.uncachedPromptTokens > 0);

  const sorted = rows.sort((x, y) =>
    discountPerToken != null
      ? (y.potentialSavings ?? 0) - (x.potentialSavings ?? 0)
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
