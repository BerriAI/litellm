import { keepPreviousData, useQuery } from "@tanstack/react-query";

import type { TimeRange } from "../api";
import { useTracesApi } from "../api";
import type { TraceHistogram } from "../types";
import { BUCKETS, type Bucket } from "./TracesTimeline";

/** The dot field draws one series entry per successful run. */
export const toBuckets = (histogram: TraceHistogram): Bucket[] =>
  histogram.buckets.map((bucket) => ({
    startMs: bucket.start_ms,
    endMs: bucket.end_ms,
    total: bucket.total,
    failed: bucket.failed,
    series: bucket.agents.flatMap(({ agent, runs }) => Array<string>(runs).fill(agent)),
  }));

const emptyBuckets = (range: TimeRange): Bucket[] => {
  const width = (range.endMs - range.startMs) / BUCKETS;
  return Array.from({ length: BUCKETS }, (_, i) => ({
    startMs: range.startMs + i * width,
    endMs: range.startMs + (i + 1) * width,
    total: 0,
    failed: 0,
    series: [],
  }));
};

/** Matching runs per bucket across the whole range, counted by the server so every run is plotted, not just loaded ones. */
export function useTraceHistogram(accessToken: string, range: TimeRange, q: string, enabled: boolean): Bucket[] {
  const traces = useTracesApi(accessToken);
  const histogram = useQuery({
    queryKey: ["agentTraceHistogram", traces.scope, range.startMs, range.endMs, q],
    queryFn: () => traces.histogram(range, q, BUCKETS),
    enabled,
    placeholderData: keepPreviousData,
    select: toBuckets,
  });
  return histogram.data ?? emptyBuckets(range);
}
