"use client";

import { ArrowDown, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";

import { StatusMark } from "./StatusMark";
import type { TraceSummary } from "./traceTypes";
import { fmtMs, previewText, traceDisplayName } from "./traceUtils";

interface AgentTracesTableProps {
  traces: TraceSummary[];
  isLoading: boolean;
  error: Error | null;
  hasMore: boolean;
  onLoadMore: () => void;
  onOpenTrace: (traceId: string) => void;
}

/** Spend is only on summaries once the spend-enrichment PR lands; show Cost when it's there. */
type SummaryWithSpend = TraceSummary & { spend?: number };

const SECOND_MS = 1000;
const MINUTE_S = 60;
const HOUR_M = 60;
const DAY_H = 24;

export function relativeTime(iso: string, now: number = Date.now()): string {
  const diffS = Math.round((now - new Date(iso).getTime()) / SECOND_MS);
  if (diffS < 5) return "just now";
  if (diffS < MINUTE_S) return `${diffS}s ago`;
  const diffM = Math.round(diffS / MINUTE_S);
  if (diffM < HOUR_M) return `${diffM}m ago`;
  const diffH = Math.round(diffM / HOUR_M);
  if (diffH < DAY_H) return `${diffH}h ago`;
  return `${Math.round(diffH / DAY_H)}d ago`;
}

export const formatCost = (cost: number): string => {
  if (cost === 0) return "$0.00";
  if (cost < 0.01) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
};

const firstLine = (text: string): string => text.split("\n")[0] ?? text;

const TH = "px-3 font-medium";
const TH_NUM = "px-3 text-right font-medium";
const TD_NUM = "px-3 text-right font-mono tabular-nums text-muted-foreground";

/** Devtool-dense runs list: one row per agent run, newest first. */
export function AgentTracesTable({
  traces,
  isLoading,
  error,
  hasMore,
  onLoadMore,
  onOpenTrace,
}: AgentTracesTableProps) {
  const showCost = traces.some((t) => typeof (t as SummaryWithSpend).spend === "number");
  const isEmpty = !isLoading && !error && traces.length === 0;
  return (
    <div className="min-h-0 flex-1 overflow-auto" data-testid="runs-table">
      <table aria-label="Agent runs" className="w-full min-w-[900px] table-fixed border-collapse text-left">
        <thead className="sticky top-0 z-sticky bg-muted/40 backdrop-blur">
          <tr className="h-8 border-b border-border text-[10px] tracking-[0.08em] text-muted-foreground uppercase">
            <th className={`w-[96px] ${TH}`}>
              <span className="inline-flex items-center gap-1">
                Time <ArrowDown className="size-2.5" />
              </span>
            </th>
            <th className={`w-[160px] ${TH}`}>Service</th>
            <th className={TH}>Input</th>
            <th className={`w-[72px] ${TH_NUM}`}>Agents</th>
            <th className={`w-[74px] ${TH_NUM}`}>Steps</th>
            <th className={`w-[86px] ${TH_NUM}`}>Duration</th>
            {showCost && <th className={`w-[80px] ${TH_NUM}`}>Cost</th>}
            <th className={`w-[72px] ${TH_NUM}`}>Failed</th>
            <th className="w-8" />
          </tr>
        </thead>
        <tbody>
          {traces.map((run) => (
            <tr
              key={run.trace_id}
              data-testid="agent-trace-row"
              onClick={() => onOpenTrace(run.trace_id)}
              className="h-9 cursor-pointer border-b border-border/60 text-[12px] hover:bg-accent/50"
            >
              <td
                className="px-3 font-mono text-[11px] tabular-nums text-muted-foreground"
                title={new Date(run.start_time).toLocaleString()}
              >
                {relativeTime(run.start_time)}
              </td>
              <td className="truncate px-3 text-muted-foreground" title={run.service}>
                {run.service}
              </td>
              <td className="px-3">
                <div className="flex min-w-0 items-center gap-2">
                  <StatusMark status={run.error_count > 0 ? "error" : "ok"} subtle />
                  <span className="truncate text-foreground">
                    {firstLine(previewText(run.input_preview)) || traceDisplayName(run)}
                  </span>
                  <span className="hidden shrink-0 font-mono text-[10px] text-muted-foreground 2xl:inline">
                    {run.trace_id}
                  </span>
                </div>
              </td>
              <td className={TD_NUM}>{run.agent_count.toLocaleString()}</td>
              <td className={TD_NUM}>{run.span_count.toLocaleString()}</td>
              <td className="px-3 text-right font-mono tabular-nums text-foreground">{fmtMs(run.duration_ms)}</td>
              {showCost && (
                <td className="px-3 text-right font-mono tabular-nums text-foreground">
                  {formatCost((run as SummaryWithSpend).spend ?? 0)}
                </td>
              )}
              <td className="px-3 text-right">
                {run.error_count > 0 ? (
                  <StatusMark status="error" count={run.error_count} />
                ) : (
                  <span className="font-mono text-[11px] text-muted-foreground/60">0</span>
                )}
              </td>
              <td>
                <ChevronRight className="size-3 text-muted-foreground/60" />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {isLoading && <div className="py-16 text-center text-[12px] text-muted-foreground">Loading runs…</div>}
      {error && (
        <div className="py-16 text-center text-[12px] text-muted-foreground">Could not load runs: {error.message}</div>
      )}
      {isEmpty && (
        <div className="py-16 text-center text-[12px] text-muted-foreground">No runs match these filters.</div>
      )}
      {hasMore && (
        <div className="border-t border-border/60 px-3 py-2">
          <Button size="xs" variant="ghost" onClick={onLoadMore}>
            Load more
          </Button>
        </div>
      )}
    </div>
  );
}
