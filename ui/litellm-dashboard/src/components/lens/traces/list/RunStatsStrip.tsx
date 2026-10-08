"use client";

import { useMemo } from "react";

import type { TimeWindow } from "@/components/shared/timeRange/timeRange";

import { StatCell, StatStrip } from "../../ui/StatStrip";
import type { TraceSummary } from "../types";
import { fmtMs } from "../utils";
import type { AgentTracesResult } from "./useAgentTraces";
import { compactNumber, formatSpend, runStats, type RunStats } from "./runStats";

function spendHint(stats: RunStats): string {
  if (stats.unpriced > 0)
    return `${stats.unpriced.toLocaleString()} ${stats.unpriced === 1 ? "run" : "runs"} not fully priced`;
  if (stats.runs === 0) return "No runs in this range";
  return `${formatSpend(stats.spend / stats.runs)} per run`;
}

export function RunStatsStrip({
  runs,
  range,
  traces,
}: {
  runs: readonly TraceSummary[];
  range: TimeWindow;
  traces: Pick<AgentTracesResult, "isLoading" | "isPlaceholder">;
}) {
  const pending = traces.isLoading || traces.isPlaceholder;
  const stats = useMemo(() => runStats(runs, range), [runs, range]);
  const ok = stats.runs - stats.failed;
  return (
    <StatStrip>
      <StatCell
        label="Runs"
        pending={pending}
        value={stats.runs.toLocaleString()}
        hint={
          <>
            <span className="text-success">{ok.toLocaleString()} ok</span>
            {" · "}
            <span className={stats.failed > 0 ? "text-destructive" : undefined}>
              {stats.failed.toLocaleString()} failed
            </span>
            {` · ${stats.agents.toLocaleString()} ${stats.agents === 1 ? "agent" : "agents"}`}
          </>
        }
        spark={stats.buckets.map((bucket) => bucket.runs)}
      />
      <StatCell
        label="Spend"
        pending={pending}
        value={formatSpend(stats.spend)}
        title={`$${stats.spend.toFixed(6)}`}
        hint={spendHint(stats)}
        spark={stats.buckets.map((bucket) => bucket.spend)}
      />
      <StatCell
        label="Duration, p50"
        pending={pending}
        value={stats.p50 === null ? "—" : fmtMs(stats.p50)}
        hint={stats.p95 === null ? "No runs in this range" : `${fmtMs(stats.p95)} p95`}
        spark={stats.buckets.map((bucket) => bucket.p50)}
      />
      <StatCell
        label="Tokens"
        pending={pending}
        value={compactNumber(stats.inputTokens + stats.outputTokens)}
        title={(stats.inputTokens + stats.outputTokens).toLocaleString()}
        hint={`${compactNumber(stats.inputTokens)} in · ${compactNumber(stats.outputTokens)} out`}
        spark={stats.buckets.map((bucket) => bucket.tokens)}
      />
    </StatStrip>
  );
}
