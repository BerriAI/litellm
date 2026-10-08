import moment from "moment";

import type { TimeWindow } from "@/components/shared/timeRange/timeRange";

import type { TraceSummary } from "../types";
import { traceAgentNames } from "../utils";

export const STAT_BUCKETS = 24;

export interface RunStatsBucket {
  readonly runs: number;
  readonly spend: number;
  readonly tokens: number;
  readonly p50: number;
}

export interface RunStats {
  readonly runs: number;
  readonly failed: number;
  readonly agents: number;
  readonly spend: number;
  readonly unpriced: number;
  readonly p50: number | null;
  readonly p95: number | null;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly buckets: readonly RunStatsBucket[];
}

const sum = (values: readonly number[]): number => values.reduce((total, value) => total + value, 0);

export function percentile(values: readonly number[], p: number): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1))];
}

const tokensOf = (run: TraceSummary): number => run.input_tokens + run.output_tokens;

function bucketOf(runs: readonly TraceSummary[]): RunStatsBucket {
  const durations = runs.map((run) => run.duration_ms);
  return {
    runs: runs.length,
    spend: sum(runs.map((run) => run.spend ?? 0)),
    tokens: sum(runs.map(tokensOf)),
    p50: percentile(durations, 50) ?? 0,
  };
}

export function bucketRunStats(
  runs: readonly TraceSummary[],
  range: TimeWindow,
  buckets = STAT_BUCKETS,
): RunStatsBucket[] {
  const width = (range.endMs - range.startMs) / buckets;
  const indexOf = (run: TraceSummary): number => Math.floor((moment(run.start_time).valueOf() - range.startMs) / width);
  return Array.from({ length: buckets }, (_, i) => bucketOf(runs.filter((run) => indexOf(run) === i)));
}

/** The span the runs actually cover, so a short burst in a long range still spreads across the sparkline. */
export function activeWindow(runs: readonly TraceSummary[], range: TimeWindow): TimeWindow {
  const starts = runs
    .map((run) => moment(run.start_time).valueOf())
    .filter((t) => t >= range.startMs && t < range.endMs);
  if (starts.length === 0) return range;
  return { startMs: Math.min(...starts), endMs: Math.max(...starts) + 1 };
}

export function runStats(runs: readonly TraceSummary[], range: TimeWindow): RunStats {
  const durations = runs.map((run) => run.duration_ms);
  return {
    runs: runs.length,
    failed: runs.filter((run) => run.status === "error").length,
    agents: new Set(runs.flatMap(traceAgentNames)).size,
    spend: sum(runs.map((run) => run.spend ?? 0)),
    unpriced: runs.filter((run) => run.spend == null || run.priced_calls < run.llm_calls).length,
    p50: percentile(durations, 50),
    p95: percentile(durations, 95),
    inputTokens: sum(runs.map((run) => run.input_tokens)),
    outputTokens: sum(runs.map((run) => run.output_tokens)),
    buckets: bucketRunStats(runs, activeWindow(runs, range)),
  };
}

export const compactNumber = (n: number): string =>
  new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(n);

export const formatSpend = (spend: number): string => {
  if (spend === 0) return "$0.00";
  if (spend < 0.01) return `$${spend.toFixed(4)}`;
  if (spend >= 1000) return `$${compactNumber(spend)}`;
  return `$${spend.toFixed(2)}`;
};
