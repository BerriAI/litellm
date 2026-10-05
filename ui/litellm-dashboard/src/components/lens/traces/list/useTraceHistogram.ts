import { keepPreviousData, useQuery } from "@tanstack/react-query";

import type { TimeWindow } from "@/components/shared/timeRange/timeRange";
import { type RunSelection, useTracesApi } from "../api";
import type { TraceHistogram } from "../types";
import { BUCKETS, type TimeBucket } from "@/components/shared/timeline/Timeline";

/** The dot field draws one series entry per successful run. */
export const toBuckets = (histogram: TraceHistogram): TimeBucket[] =>
  histogram.buckets.map((bucket) => ({
    startMs: bucket.start_ms,
    endMs: bucket.end_ms,
    total: bucket.total,
    failed: bucket.failed,
    series: bucket.agents.flatMap(({ agent, runs }) => Array<string>(runs).fill(agent)),
  }));

const emptyBuckets = (range: TimeWindow): TimeBucket[] => {
  const width = (range.endMs - range.startMs) / BUCKETS;
  return Array.from({ length: BUCKETS }, (_, i) => ({
    startMs: range.startMs + i * width,
    endMs: range.startMs + (i + 1) * width,
    total: 0,
    failed: 0,
    series: [],
  }));
};

export interface TraceHistogramResult {
  buckets: TimeBucket[];
  /** True until the first histogram for this scope arrives; a range change keeps the previous one instead. */
  isLoading: boolean;
}

/** Matching runs per bucket across the whole range, counted by the server so every run is plotted, not just loaded ones. */
export function useTraceHistogram(
  accessToken: string,
  selection: RunSelection,
  enabled: boolean,
): TraceHistogramResult {
  const traces = useTracesApi(accessToken);
  const { window, q } = selection;
  const histogram = useQuery({
    queryKey: ["agentTraceHistogram", traces.scope, window.startMs, window.endMs, q],
    queryFn: () => traces.histogram(selection, BUCKETS),
    enabled,
    placeholderData: keepPreviousData,
    select: toBuckets,
  });
  return { buckets: histogram.data ?? emptyBuckets(window), isLoading: histogram.isLoading };
}
