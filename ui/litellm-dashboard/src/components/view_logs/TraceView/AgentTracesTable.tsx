"use client";

import { ArrowDown, ChevronRight } from "lucide-react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useEffect, useState } from "react";
import { useInView } from "react-intersection-observer";

import { PANEL_TRIGGER } from "@/components/shared/SidePanel";
import { Button } from "@/components/ui/button";
import { formatActivityTimestamp, formatRunTimestamp, localTimeZoneAbbreviation } from "@/utils/activityTimestamp";
import { cn } from "@/lib/cva.config";

import { SpanIcon } from "./SpanIcon";
import { StatusMark } from "./StatusMark";
import { FrameworkLogo, traceFramework } from "./TraceFramework";
import type { TraceSummary } from "./traceTypes";
import { fmtMs, previewText, traceDisplayName, traceAgentNames } from "./traceUtils";

interface AgentTracesTableProps {
  traces: TraceSummary[];
  isLoading: boolean;
  error: Error | null;
  hasMore: boolean;
  isFetching?: boolean;
  onRetry?: () => void;
  onLoadMore: () => void;
  onOpenTrace: (trace: TraceSummary) => void;
  selectedKey?: string | null;
  rangeEmpty?: boolean;
  onSetUpTracing: () => void;
}

export const formatCost = (cost: number): string => {
  if (cost === 0) return "$0.00";
  if (cost < 0.01) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
};

const firstLine = (text: string): string => text.split("\n")[0] ?? text;
const runKey = (run: TraceSummary): string => run.trace_ref || run.trace_id;

const TH = "px-3 font-medium";
const PREFETCH_MARGIN = "0px 0px 480px 0px";
const PLACEHOLDER_ROWS = [0, 1, 2];
const ROW_HEIGHT = 36;
const OVERSCAN_ROWS = 12;
const TH_NUM = "px-3 text-right font-medium";
const TD_NUM = "px-3 text-right font-mono tabular-nums text-muted-foreground";

function AgentCell({ run }: { run: TraceSummary }) {
  const framework = traceFramework(run);
  const agents = traceAgentNames(run).join(", ");
  const title = [agents, framework?.label].filter(Boolean).join(" · ");
  return (
    <td className="px-3 text-muted-foreground" title={title}>
      <div className="flex min-w-0 items-center gap-1.5">
        {framework ? <FrameworkLogo framework={framework} /> : <SpanIcon type="agent" size="sm" />}
        <span className="truncate">{agents || framework?.label || "—"}</span>
      </div>
    </td>
  );
}

function PlaceholderRow({ rowRef }: { rowRef?: (node: Element | null) => void }) {
  return (
    <tr ref={rowRef} aria-hidden data-testid="runs-placeholder" className="h-9 border-b border-border/60">
      <td className="px-3">
        <div className="h-2.5 w-28 animate-pulse rounded-sm bg-muted motion-reduce:animate-none" />
      </td>
      <td className="px-3">
        <div className="h-2.5 w-24 animate-pulse rounded-sm bg-muted motion-reduce:animate-none" />
      </td>
      <td className="px-3" colSpan={7}>
        <div className="h-2.5 w-2/5 animate-pulse rounded-sm bg-muted motion-reduce:animate-none" />
      </td>
    </tr>
  );
}

function EmptyRuns({ rangeEmpty, onSetUpTracing }: { rangeEmpty: boolean; onSetUpTracing: () => void }) {
  if (!rangeEmpty)
    return (
      <div className="flex items-center justify-center gap-3 py-16 text-xs text-muted-foreground">
        <span>No runs match these filters.</span>
        <Button size="xs" variant="outline" onClick={onSetUpTracing}>
          Set up tracing
        </Button>
      </div>
    );
  return (
    <div className="flex flex-col items-center gap-1 py-16 text-center">
      <p className="text-sm font-medium">No runs in this time range</p>
      <p className="text-xs text-muted-foreground">Connect an agent to start sending traces.</p>
      <Button size="sm" className="mt-3" onClick={onSetUpTracing}>
        Set up tracing
      </Button>
    </div>
  );
}

function RunRow({
  run,
  selected,
  onOpen,
}: {
  run: TraceSummary;
  selected: boolean;
  onOpen: (run: TraceSummary) => void;
}) {
  return (
    <tr
      data-testid="agent-trace-row"
      {...PANEL_TRIGGER}
      onClick={() => onOpen(run)}
      aria-selected={selected}
      className={cn(
        "h-9 cursor-pointer border-b border-border/60 text-xs transition-colors duration-150 motion-reduce:transition-none",
        selected ? "bg-trace-row-selected shadow-[inset_2px_0_0_var(--trace-brand)]" : "hover:bg-trace-row-hover",
      )}
    >
      <td
        className="px-3 font-mono text-xs whitespace-nowrap tabular-nums text-muted-foreground"
        title={formatActivityTimestamp(run.start_time)}
      >
        {formatRunTimestamp(run.start_time)}
      </td>
      <AgentCell run={run} />
      <td className="px-3">
        <div className="flex min-w-0 items-center gap-2">
          <StatusMark status={run.error_count > 0 ? "error" : "ok"} subtle />
          <span className="truncate text-foreground">
            {firstLine(previewText(run.input_preview)) || traceDisplayName(run)}
          </span>
          {run.resolution_limited && (
            <span
              className="shrink-0 text-xs text-muted-foreground"
              title="This run is too large to calculate all totals in this view"
            >
              Partial totals
            </span>
          )}
          <span className="hidden shrink-0 font-mono text-xs text-muted-foreground 2xl:inline">{run.trace_id}</span>
        </div>
      </td>
      <td className={TD_NUM}>{run.agent_count.toLocaleString()}</td>
      <td className={TD_NUM}>{run.span_count.toLocaleString()}</td>
      <td className="px-3 text-right font-mono tabular-nums text-foreground">{fmtMs(run.duration_ms)}</td>
      <td className="px-3 text-right font-mono tabular-nums text-foreground">
        {run.spend == null ? "—" : formatCost(run.spend)}
      </td>
      <td className="px-3 text-right">
        {run.error_count > 0 ? (
          <StatusMark status="error" count={run.error_count} />
        ) : (
          <span className="font-mono text-xs text-muted-foreground/60">0</span>
        )}
      </td>
      <td>
        <ChevronRight className="size-3 text-muted-foreground/60" />
      </td>
    </tr>
  );
}

function useVirtualRows(traces: TraceSummary[], scroller: HTMLDivElement | null) {
  const options = {
    count: traces.length,
    getScrollElement: () => scroller,
    estimateSize: () => ROW_HEIGHT,
    overscan: OVERSCAN_ROWS,
    getItemKey: (index: number) => runKey(traces[index]),
  };
  const virtualizer = useVirtualizer(options);
  const rows = virtualizer.getVirtualItems();
  const padTop = rows[0]?.start ?? 0;
  const padBottom = virtualizer.getTotalSize() - (rows.at(-1)?.end ?? 0);
  return { rows, padTop, padBottom };
}

/** Devtool-dense runs list: one row per agent run, newest first. */
export function AgentTracesTable({
  traces,
  isLoading,
  error,
  hasMore,
  isFetching = false,
  onRetry,
  onLoadMore,
  onOpenTrace,
  selectedKey = null,
  rangeEmpty = false,
  onSetUpTracing,
}: AgentTracesTableProps) {
  const settled = !isLoading && !error;
  const isEmpty = settled && !hasMore && traces.length === 0;
  const canContinue = settled && hasMore;
  const autoContinue = canContinue && traces.length > 0;
  const [scroller, setScroller] = useState<HTMLDivElement | null>(null);
  const { ref: tailRef, inView: nearTail } = useInView({ root: scroller, rootMargin: PREFETCH_MARGIN });
  useEffect(() => {
    if (nearTail && autoContinue && !isFetching) onLoadMore();
  }, [nearTail, autoContinue, isFetching, onLoadMore]);
  const { rows, padTop, padBottom } = useVirtualRows(traces, scroller);
  return (
    <div ref={setScroller} className="min-h-0 flex-1 overflow-auto" data-testid="runs-table">
      <table
        aria-label="Agent runs"
        aria-busy={isFetching}
        className="w-full min-w-[900px] table-fixed border-collapse text-left"
      >
        <thead className="sticky top-0 z-sticky bg-[color-mix(in_oklab,var(--muted)_40%,var(--card))]">
          <tr className="h-8 border-b border-border text-xs tracking-wider text-muted-foreground uppercase">
            <th className={`w-[170px] ${TH}`}>
              <span className="inline-flex items-center gap-1 whitespace-nowrap">
                Time <ArrowDown className="size-2.5" />
                <span className="font-normal normal-case tracking-normal text-muted-foreground/70">
                  {localTimeZoneAbbreviation()}
                </span>
              </span>
            </th>
            <th className={`w-[160px] ${TH}`}>Agent</th>
            <th className={TH}>Input</th>
            <th className={`w-[72px] ${TH_NUM}`}>Agents</th>
            <th className={`w-[74px] ${TH_NUM}`}>Steps</th>
            <th className={`w-[86px] ${TH_NUM}`}>Duration</th>
            <th className={`w-[80px] ${TH_NUM}`}>Cost</th>
            <th className={`w-[72px] ${TH_NUM}`}>Failed</th>
            <th className="w-8" />
          </tr>
        </thead>
        <tbody>
          {padTop > 0 && <tr aria-hidden style={{ height: padTop }} />}
          {rows.map(({ index, key }) => (
            <RunRow key={key} run={traces[index]} selected={selectedKey === key} onOpen={onOpenTrace} />
          ))}
          {padBottom > 0 && <tr aria-hidden style={{ height: padBottom }} />}
          {autoContinue &&
            PLACEHOLDER_ROWS.map((row) => <PlaceholderRow key={row} rowRef={row === 0 ? tailRef : undefined} />)}
        </tbody>
      </table>
      {isLoading && <div className="py-16 text-center text-xs text-muted-foreground">Loading runs…</div>}
      {error && (
        <div role="alert" className="flex items-center justify-center gap-3 py-6 text-xs text-muted-foreground">
          <span>
            {traces.length ? "Could not load more runs" : "Could not load runs"}: {error.message}
          </span>
          {onRetry && (
            <Button size="xs" variant="outline" disabled={isFetching} onClick={onRetry}>
              Retry
            </Button>
          )}
        </div>
      )}
      {canContinue && traces.length === 0 && (
        <div className="flex items-center justify-center gap-3 py-16 text-xs text-muted-foreground">
          <span>No loaded runs match these filters.</span>
          <Button size="xs" variant="outline" disabled={isFetching} onClick={onLoadMore}>
            Load older runs
          </Button>
        </div>
      )}
      {isEmpty && <EmptyRuns rangeEmpty={rangeEmpty} onSetUpTracing={onSetUpTracing} />}
    </div>
  );
}
