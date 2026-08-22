import { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import { ToolSpendDailyEntry, ToolSpendEntry } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export const usd = (value: number): string => {
  // Sized and signed off the magnitude: a driver can come out negative, and a small
  // loss rendered at two decimals would read as "$-0.00"
  const magnitude = Math.abs(value);
  const decimals = magnitude > 0 && magnitude < 1 ? 4 : 2;
  return `${value < 0 ? "-" : ""}$${formatNumberWithCommas(magnitude, decimals)}`;
};

export const pct = (ratio: number): string => `${formatNumberWithCommas(ratio * 100, 1)}%`;

export const shortDate = (iso: string): string =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });

export const compressionOf = (m: SpendMetrics): number => m.compression_savings_spend ?? 0;
export const cachingOf = (m: SpendMetrics): number => m.prompt_caching_savings_spend ?? 0;
export const autorouterOf = (m: SpendMetrics): number => m.autorouter_savings_spend ?? 0;
export const savedTokensOf = (m: SpendMetrics): number => m.compression_saved_tokens ?? 0;

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
  netSavingsPerCachedToken: number | null;
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
      cachedTokens: agg.cachedTokens + a.cacheReadTokens + a.cacheCreationTokens,
      realizedCachingSavings: agg.realizedCachingSavings + a.realizedCachingSavings,
    }),
    { cachedTokens: 0, realizedCachingSavings: 0 },
  );
  // prompt_caching_savings_spend is net of the cache-write premium, so the rate has to
  // divide by every token that took the cache path -- a key that starts caching pays
  // those write premiums too. Dividing by reads alone overstates it and, on write-heavy
  // traffic where the net is negative, would flip the sign of a real loss into a saving
  const netSavingsPerCachedToken = totals.cachedTokens > 0 ? totals.realizedCachingSavings / totals.cachedTokens : null;
  // A non-positive rate prices no leakage: there is no saving to extrapolate from
  const rate = netSavingsPerCachedToken != null && netSavingsPerCachedToken > 0 ? netSavingsPerCachedToken : null;

  const rows: CacheLeakageRow[] = [...byEntity.entries()]
    .map(([id, a]) => {
      const uncachedPromptTokens = Math.max(0, a.promptTokens - a.cacheReadTokens - a.cacheCreationTokens);
      return {
        id,
        label: dimension === "model" ? id : a.alias ?? `${id.slice(0, 8)}...`,
        sublabel: dimension === "model" ? null : a.teamId,
        uncachedPromptTokens,
        cacheHitRatio: a.promptTokens > 0 ? a.cacheReadTokens / a.promptTokens : 0,
        potentialSavings: rate != null ? uncachedPromptTokens * rate : null,
      };
    })
    .filter((row) => row.uncachedPromptTokens > 0);

  const sorted = rows.sort((x, y) =>
    rate != null
      ? (y.potentialSavings ?? 0) - (x.potentialSavings ?? 0)
      : y.uncachedPromptTokens - x.uncachedPromptTokens,
  );

  return { rows: sorted.slice(0, limit), netSavingsPerCachedToken };
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

export const localIsoDay = (d: Date): string =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;

export type SavingsAccumulation = "cumulative" | "per-interval";

// A type alias, not an interface: only aliases get the implicit index signature
// that the chart wrappers' `Record<string, unknown>` datum bound requires.
export type SavingsPoint = {
  date: string;
  Compression: number;
  "Prompt caching": number;
  "Auto-router": number;
};

/**
 * The savings drivers, each owning its own colour.
 *
 * One list rather than a names list beside a colours list, because the donut is
 * given only the drivers that saved anything and charts assign colours by position
 * in the data they receive. Two lists that line up by index therefore stop lining
 * up the moment a driver is filtered out: the survivors slide down and inherit the
 * colours of the drivers above them, while the legend still reports the original
 * mapping. Colour travels with the driver so filtering cannot separate them.
 */
export const SAVINGS_DRIVERS = [
  { name: "Compression", color: "emerald" },
  { name: "Prompt caching", color: "blue" },
  { name: "Auto-router", color: "amber" },
] as const;

export const SAVINGS_SERIES = SAVINGS_DRIVERS.map((d) => d.name);
export const SAVINGS_COLORS = SAVINGS_DRIVERS.map((d) => d.color);

/**
 * Running total of each series across the selected window. The total restarts
 * at the beginning of the range rather than carrying in earlier spend, which is
 * what "running total saved, <range>" claims on the card.
 */
export const toCumulative = (points: readonly SavingsPoint[]): SavingsPoint[] =>
  points.reduce<SavingsPoint[]>((acc, point) => {
    const previous = acc[acc.length - 1];
    return [
      ...acc,
      {
        date: point.date,
        Compression: (previous?.Compression ?? 0) + point.Compression,
        "Prompt caching": (previous?.["Prompt caching"] ?? 0) + point["Prompt caching"],
        "Auto-router": (previous?.["Auto-router"] ?? 0) + point["Auto-router"],
      },
    ];
  }, []);

/**
 * Prepend a synthetic $0 point at the start of the range so the cumulative line
 * rises from zero instead of floating as a single dot. The daily rollup only
 * resolves whole days, so a one-day range would otherwise be one point; with the
 * anchor it reads as "start of range $0 climbing to the range's running total".
 * An empty series is left untouched so the chart's own "No data" state shows.
 */
export const withStartAnchor = (cumulative: readonly SavingsPoint[], startLabel: string): SavingsPoint[] =>
  cumulative.length === 0
    ? [...cumulative]
    : [{ date: startLabel, Compression: 0, "Prompt caching": 0, "Auto-router": 0 }, ...cumulative];

/** "Jul 16 – Jul 23", collapsing to a single date when the range is one day. */
export const formatRangeLabel = (from: Date | undefined, to: Date | undefined): string => {
  if (!from || !to) return "";
  const short = (d: Date) => d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  const start = short(from);
  const end = short(to);
  return start === end ? start : `${start} – ${end}`;
};

/**
 * Dots mark each reading, as in the design. Past this many readings they crowd
 * into a solid band and stop being readable, so the line carries it alone.
 */
export const MAX_POINTS_WITH_DOTS = 31;
