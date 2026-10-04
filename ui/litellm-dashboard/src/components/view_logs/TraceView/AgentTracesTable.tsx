"use client";

import { getCoreRowModel, useReactTable, type ColumnDef, type TableOptions } from "@tanstack/react-table";
import { ArrowDown, ChevronRight } from "lucide-react";
import { useEffect } from "react";
import { useInView } from "react-intersection-observer";

import { InspectorTable, useInspectorTable } from "@/components/shared/InspectorTable";
import { Button } from "@/components/ui/button";
import { formatActivityTimestamp, formatRunTimestamp, localTimeZoneAbbreviation } from "@/utils/activityTimestamp";

import { SpanIcon } from "./SpanIcon";
import { StatusMark } from "./StatusMark";
import { FrameworkLogo, traceFramework } from "./TraceFramework";
import type { TraceSummary } from "./traceTypes";
import { traceRefOf } from "./traceRouting";
import { fmtMs, previewText, traceDisplayName, traceAgentNames } from "./traceUtils";

interface AgentTracesTableProps {
  traces: TraceSummary[];
  isLoading: boolean;
  error: Error | null;
  hasMore: boolean;
  isFetching?: boolean;
  onRetry?: () => void;
  onLoadMore: () => void;
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

const PREFETCH_MARGIN = "0px 0px 480px 0px";
const PLACEHOLDER_ROWS = [0, 1, 2];
const ROW_HEIGHT = 36;
const MUTED_NUM = "font-mono text-muted-foreground";
const NUM = "font-mono text-foreground";

function AgentCell({ run }: { run: TraceSummary }) {
  const framework = traceFramework(run);
  const agents = traceAgentNames(run).join(", ");
  const title = [agents, framework?.label].filter(Boolean).join(" · ");
  return (
    <div className="flex min-w-0 items-center gap-1.5 text-muted-foreground" title={title}>
      {framework ? <FrameworkLogo framework={framework} /> : <SpanIcon type="agent" size="sm" />}
      <span className="truncate">{agents || framework?.label || "—"}</span>
    </div>
  );
}

function InputCell({ run }: { run: TraceSummary }) {
  return (
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
  );
}

const RUN_COLUMNS: ColumnDef<TraceSummary>[] = [
  {
    id: "time",
    size: 170,
    header: () => (
      <span className="inline-flex items-center gap-1 whitespace-nowrap">
        Time <ArrowDown className="size-2.5" />
        <span className="font-normal normal-case tracking-normal text-muted-foreground/70">
          {localTimeZoneAbbreviation()}
        </span>
      </span>
    ),
    cell: ({ row }) => (
      <span title={formatActivityTimestamp(row.original.start_time)}>
        {formatRunTimestamp(row.original.start_time)}
      </span>
    ),
    meta: { className: "font-mono tabular-nums text-muted-foreground" },
  },
  { id: "agent", size: 160, header: "Agent", cell: ({ row }) => <AgentCell run={row.original} /> },
  { id: "input", header: "Input", cell: ({ row }) => <InputCell run={row.original} /> },
  {
    id: "agents",
    size: 72,
    header: "Agents",
    cell: ({ row }) => row.original.agent_count.toLocaleString(),
    meta: { numeric: true, className: MUTED_NUM },
  },
  {
    id: "steps",
    size: 74,
    header: "Steps",
    cell: ({ row }) => row.original.span_count.toLocaleString(),
    meta: { numeric: true, className: MUTED_NUM },
  },
  {
    id: "duration",
    size: 86,
    header: "Duration",
    cell: ({ row }) => fmtMs(row.original.duration_ms),
    meta: { numeric: true, className: NUM },
  },
  {
    id: "cost",
    size: 80,
    header: "Cost",
    cell: ({ row }) => (row.original.spend == null ? "—" : formatCost(row.original.spend)),
    meta: { numeric: true, className: NUM },
  },
  {
    id: "failed",
    size: 72,
    header: "Failed",
    cell: ({ row }) =>
      row.original.error_count > 0 ? (
        <StatusMark status="error" count={row.original.error_count} />
      ) : (
        <span className="font-mono text-muted-foreground/60">0</span>
      ),
    meta: { numeric: true },
  },
  {
    id: "open",
    size: 32,
    header: "",
    cell: () => <ChevronRight className="size-3 text-muted-foreground/60" />,
    meta: { className: "px-0" },
  },
];

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

function LoadMoreRows({ isFetching, onLoadMore }: { isFetching: boolean; onLoadMore: () => void }) {
  const { scroller } = useInspectorTable();
  const { ref: tailRef, inView: nearTail } = useInView({ root: scroller, rootMargin: PREFETCH_MARGIN });
  useEffect(() => {
    if (nearTail && !isFetching) onLoadMore();
  }, [nearTail, isFetching, onLoadMore]);
  return PLACEHOLDER_ROWS.map((row) => <PlaceholderRow key={row} rowRef={row === 0 ? tailRef : undefined} />);
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

/** Devtool-dense runs list: one row per agent run, newest first. */
export function AgentTracesTable({
  traces,
  isLoading,
  error,
  hasMore,
  isFetching = false,
  onRetry,
  onLoadMore,
  rangeEmpty = false,
  onSetUpTracing,
}: AgentTracesTableProps) {
  const settled = !isLoading && !error;
  const isEmpty = settled && !hasMore && traces.length === 0;
  const canContinue = settled && hasMore;
  const autoContinue = canContinue && traces.length > 0;
  const tableOptions: TableOptions<TraceSummary> = {
    data: traces,
    columns: RUN_COLUMNS,
    getRowId: runKey,
    autoResetAll: false,
    getCoreRowModel: getCoreRowModel(),
  };
  const table = useReactTable(tableOptions);
  return (
    <InspectorTable.Root table={table} data-testid="runs-table">
      <InspectorTable.Grid aria-label="Agent runs" aria-busy={isFetching} className="min-w-[900px] text-xs">
        <InspectorTable.Header />
        <InspectorTable.Body<TraceSummary>
          rowHeight={() => ROW_HEIGHT}
          after={autoContinue && <LoadMoreRows isFetching={isFetching} onLoadMore={onLoadMore} />}
        >
          {(row) => (
            <InspectorTable.Row
              row={row}
              item={traceRefOf(row.original)}
              data-testid="agent-trace-row"
              className="h-9"
            />
          )}
        </InspectorTable.Body>
      </InspectorTable.Grid>
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
    </InspectorTable.Root>
  );
}
