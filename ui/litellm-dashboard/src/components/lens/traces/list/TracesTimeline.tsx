"use client";

import moment from "moment";
import { useMemo } from "react";

import { Timeline, TIMELINE_BUCKETS, type Bucket, type TimeWindow } from "@/components/shared/timeline/Timeline";

import type { TraceSummary } from "../types";
import { traceAgentNames } from "../utils";

/** Run counts per equal-width time bucket across the window; runs outside it are dropped. */
export function bucketRuns(runs: readonly TraceSummary[], range: TimeWindow, buckets = TIMELINE_BUCKETS): Bucket[] {
  const width = (range.endMs - range.startMs) / buckets;
  const placed = runs.map((run) => ({
    index: Math.floor((moment(run.start_time).valueOf() - range.startMs) / width),
    failed: run.status === "error",
    agent: traceAgentNames(run)[0] ?? "",
  }));
  return Array.from({ length: buckets }, (_, i) => {
    const hits = placed.filter((p) => p.index === i);
    return {
      startMs: range.startMs + i * width,
      endMs: range.startMs + (i + 1) * width,
      total: hits.length,
      failed: hits.filter((p) => p.failed).length,
      series: hits.filter((p) => !p.failed).map((p) => p.agent),
    };
  });
}

interface TracesTimelineProps {
  runs: readonly TraceSummary[];
  range: TimeWindow;
  selection: TimeWindow | null;
  onSelect: (selection: TimeWindow | null) => void;
}

/** Lens adapter: buckets the currently loaded runs and hands them to the shared timeline. */
export function TracesTimeline({ runs, range, selection, onSelect }: TracesTimelineProps) {
  const buckets = useMemo(() => bucketRuns(runs, range), [runs, range]);
  return <Timeline buckets={buckets} selection={selection} onSelect={onSelect} />;
}
