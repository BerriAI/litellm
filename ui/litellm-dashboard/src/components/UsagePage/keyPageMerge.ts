import type { BreakdownMetrics, DailyData, MetricWithMetadata } from "./types";

type AggregatedResponse<M> = {
  results: DailyData[];
  metadata: M;
};

const BREAKDOWN_LISTS = ["models", "model_groups", "providers", "mcp_servers"] as const;

type EntryMap = { [key: string]: MetricWithMetadata };

function mergeEntryMap(prev: EntryMap | undefined, next: EntryMap | undefined): EntryMap {
  const entries = Object.entries(next ?? {}).map(([name, nextEntry]) => {
    const prevEntry = prev?.[name];
    return [
      name,
      prevEntry
        ? { ...prevEntry, api_key_breakdown: { ...prevEntry.api_key_breakdown, ...nextEntry.api_key_breakdown } }
        : nextEntry,
    ] as const;
  });
  return { ...prev, ...Object.fromEntries(entries) };
}

function mergeDay(prev: DailyData, next: DailyData): DailyData {
  const breakdown: BreakdownMetrics = {
    ...prev.breakdown,
    api_keys: { ...prev.breakdown.api_keys, ...next.breakdown.api_keys },
    ...Object.fromEntries(
      BREAKDOWN_LISTS.map((list) => [list, mergeEntryMap(prev.breakdown[list], next.breakdown[list])] as const),
    ),
    ...(prev.breakdown.endpoints !== undefined || next.breakdown.endpoints !== undefined
      ? { endpoints: mergeEntryMap(prev.breakdown.endpoints, next.breakdown.endpoints) }
      : {}),
  };
  return { ...prev, breakdown };
}

export function mergeKeyPage<M extends { next_cursor?: string | null }>(
  prev: AggregatedResponse<M>,
  next: AggregatedResponse<M>,
): AggregatedResponse<M> {
  const nextByDate = new Map(next.results.map((day) => [day.date, day] as const));
  const results = [
    ...prev.results.map((day) => {
      const nextDay = nextByDate.get(day.date);
      return nextDay ? mergeDay(day, nextDay) : day;
    }),
    ...next.results.filter((nextDay) => !prev.results.some((day) => day.date === nextDay.date)),
  ];
  return {
    results,
    metadata: { ...prev.metadata, next_cursor: next.metadata?.next_cursor ?? null },
  };
}
