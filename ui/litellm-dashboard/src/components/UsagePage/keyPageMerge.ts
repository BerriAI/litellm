import type { DailyData, MetricWithMetadata } from "./types";

type AggregatedResponse<M> = {
  results: DailyData[];
  metadata: M;
};

const BREAKDOWN_LISTS = ["models", "model_groups", "providers", "mcp_servers"] as const;

function mergeEntryMap(
  prev: { [key: string]: MetricWithMetadata } | undefined,
  next: { [key: string]: MetricWithMetadata } | undefined,
): { [key: string]: MetricWithMetadata } {
  const merged: { [key: string]: MetricWithMetadata } = { ...prev };
  for (const [name, nextEntry] of Object.entries(next ?? {})) {
    const prevEntry = merged[name];
    merged[name] = prevEntry
      ? { ...prevEntry, api_key_breakdown: { ...prevEntry.api_key_breakdown, ...nextEntry.api_key_breakdown } }
      : nextEntry;
  }
  return merged;
}

function mergeDay(prev: DailyData, next: DailyData): DailyData {
  const breakdown = { ...prev.breakdown };
  breakdown.api_keys = { ...prev.breakdown.api_keys, ...next.breakdown.api_keys };
  for (const list of BREAKDOWN_LISTS) {
    breakdown[list] = mergeEntryMap(prev.breakdown[list], next.breakdown[list]);
  }
  if (prev.breakdown.endpoints !== undefined || next.breakdown.endpoints !== undefined) {
    breakdown.endpoints = mergeEntryMap(prev.breakdown.endpoints, next.breakdown.endpoints);
  }
  return { ...prev, breakdown };
}

export function mergeKeyPage<M extends { next_cursor?: string | null }>(
  prev: AggregatedResponse<M>,
  next: AggregatedResponse<M>,
): AggregatedResponse<M> {
  const mergedResults = [...prev.results];
  for (const nextDay of next.results) {
    const index = mergedResults.findIndex((day) => day.date === nextDay.date);
    if (index === -1) {
      mergedResults.push(nextDay);
    } else {
      mergedResults[index] = mergeDay(mergedResults[index], nextDay);
    }
  }
  return {
    results: mergedResults,
    metadata: { ...prev.metadata, next_cursor: next.metadata?.next_cursor ?? null },
  };
}
