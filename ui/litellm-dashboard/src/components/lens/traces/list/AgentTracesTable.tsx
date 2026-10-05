"use client";

import {
  getCoreRowModel,
  useReactTable,
  type ColumnDef,
  type HeaderContext,
  type TableOptions,
} from "@tanstack/react-table";
import { ChevronRight } from "lucide-react";
import { useEffect, useMemo, type ReactNode } from "react";
import { useInView } from "react-intersection-observer";

import { DataTableSortHeader } from "@/components/shared/DataTable/DataTableSortHeader";
import { DataTableViewOptions } from "@/components/shared/DataTable/DataTableViewOptions";
import { usePersistedColumnVisibility } from "@/components/shared/DataTable/usePersistedColumnVisibility";
import { InspectorTable, useInspectorTable } from "@/components/shared/InspectorTable";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cva.config";
import { formatActivityTimestamp, formatRunTimestamp, localTimeZoneAbbreviation } from "@/utils/activityTimestamp";

import { SpanIcon } from "../ui/SpanIcon";
import { StatusMark } from "../ui/StatusMark";
import { FrameworkLogo, traceFramework } from "../ui/TraceFramework";
import type { TraceSummary } from "../types";
import { traceRefOf } from "../routing";
import { fmtMs, previewText, traceDisplayName, traceAgentNames } from "../utils";
import { fromSorting, NEWEST, type RunOrder, toSorting } from "./runOrder";

interface AgentTracesTableProps {
  traces: TraceSummary[];
  isLoading: boolean;
  error: Error | null;
  hasMore: boolean;
  isFetching?: boolean;
  /** Rows belong to the previous order, search or window while this one loads. */
  isPlaceholder?: boolean;
  /** The order rows arrive in; with `onOrderChange`, the sortable headers change it on the server. */
  order?: RunOrder;
  onOrderChange?: (order: RunOrder) => void;
  onRetry?: () => void;
  onLoadMore: () => void;
  rangeEmpty?: boolean;
  onSetUpTracing: () => void;
  /** When set, a leading checkbox column picks individual runs. */
  picks?: RunPicks;
}

export interface RunPicks {
  readonly isPicked: (run: TraceSummary) => boolean;
  readonly toggle: (run: TraceSummary, picked: boolean) => void;
}

const pickColumn = (picks: RunPicks): ColumnDef<TraceSummary> => ({
  id: "pick",
  size: 36,
  enableHiding: false,
  enableSorting: false,
  header: () => null,
  cell: ({ row }) => (
    <input
      type="checkbox"
      aria-label={`Select ${traceDisplayName(row.original)}`}
      checked={picks.isPicked(row.original)}
      onClick={(event) => event.stopPropagation()}
      onChange={(event) => picks.toggle(row.original, event.target.checked)}
    />
  ),
  meta: { title: "Pick" },
});

export const formatCost = (cost: number): string => {
  if (cost === 0) return "$0.00";
  if (cost < 0.01) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
};

const firstLine = (text: string): string => text.split("\n")[0] ?? text;
const runKey = (run: TraceSummary): string => run.trace_ref || run.trace_id;

const PREFETCH_MARGIN = "0px 0px 480px 0px";
const PLACEHOLDER_ROWS = [0, 1, 2];
const SKELETON_ROWS = Array.from({ length: 12 }, (_, i) => i);
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

const sortHeader = (title: ReactNode) =>
  function SortHeaderCell({ column }: HeaderContext<TraceSummary, unknown>) {
    return (
      <DataTableSortHeader
        column={column}
        title={title}
        className={cn("w-full font-normal uppercase", column.columnDef.meta?.numeric && "justify-end")}
      />
    );
  };

const TIME_TITLE = (
  <span className="inline-flex items-center gap-1 whitespace-nowrap">
    Time
    <span className="normal-case tracking-normal text-muted-foreground/70">{localTimeZoneAbbreviation()}</span>
  </span>
);

const RUN_COLUMNS: ColumnDef<TraceSummary>[] = [
  {
    id: "start_ms",
    accessorKey: "start_time",
    enableSorting: true,
    size: 170,
    enableHiding: false,
    header: sortHeader(TIME_TITLE),
    cell: ({ row }) => (
      <span title={formatActivityTimestamp(row.original.start_time)}>
        {formatRunTimestamp(row.original.start_time)}
      </span>
    ),
    meta: {
      title: "Time",
      className: "font-mono tabular-nums text-muted-foreground",
      renderSkeleton: () => <Skeleton className="h-3 w-24" />,
    },
  },
  {
    id: "agent",
    size: 160,
    enableHiding: false,
    header: "Agent",
    cell: ({ row }) => <AgentCell run={row.original} />,
    meta: {
      renderSkeleton: () => (
        <div className="flex items-center gap-1.5">
          <Skeleton className="size-3.5 rounded-full" />
          <Skeleton className="h-3 w-20" />
        </div>
      ),
    },
  },
  { id: "input", header: "Input", cell: ({ row }) => <InputCell run={row.original} /> },
  {
    id: "agents",
    size: 72,
    header: "Agents",
    cell: ({ row }) => row.original.agent_count.toLocaleString(),
    meta: { numeric: true, className: MUTED_NUM },
  },
  {
    id: "span_count",
    accessorKey: "span_count",
    enableSorting: true,
    size: 74,
    header: sortHeader("Steps"),
    cell: ({ row }) => row.original.span_count.toLocaleString(),
    meta: { title: "Steps", numeric: true, className: MUTED_NUM },
  },
  {
    id: "duration_ms",
    accessorKey: "duration_ms",
    enableSorting: true,
    size: 86,
    header: sortHeader("Duration"),
    cell: ({ row }) => fmtMs(row.original.duration_ms),
    meta: { title: "Duration", numeric: true, className: NUM },
  },
  {
    id: "cost",
    size: 80,
    header: "Cost",
    cell: ({ row }) => (row.original.spend == null ? "—" : formatCost(row.original.spend)),
    meta: { numeric: true, className: NUM },
  },
  {
    id: "error_count",
    accessorKey: "error_count",
    enableSorting: true,
    size: 72,
    header: sortHeader("Failed"),
    cell: ({ row }) =>
      row.original.error_count > 0 ? (
        <StatusMark status="error" count={row.original.error_count} />
      ) : (
        <span className="font-mono text-muted-foreground/60">0</span>
      ),
    meta: { title: "Failed", numeric: true },
  },
  {
    id: "open",
    size: 32,
    enableHiding: false,
    header: ({ table }) => <DataTableViewOptions table={table} label="Columns" iconOnly />,
    cell: () => <ChevronRight className="size-3 text-muted-foreground/60" />,
    meta: { className: "px-0", headerClassName: "px-1", renderSkeleton: () => null },
  },
];

function PlaceholderRow({ index, rowRef }: { index: number; rowRef?: (node: Element | null) => void }) {
  return <InspectorTable.SkeletonRow ref={rowRef} index={index} data-testid="runs-placeholder" className="h-9" />;
}

function LoadMoreRows({ isFetching, onLoadMore }: { isFetching: boolean; onLoadMore: () => void }) {
  const { scroller } = useInspectorTable();
  const { ref: tailRef, inView: nearTail } = useInView({ root: scroller, rootMargin: PREFETCH_MARGIN });
  useEffect(() => {
    if (nearTail && !isFetching) onLoadMore();
  }, [nearTail, isFetching, onLoadMore]);
  return PLACEHOLDER_ROWS.map((row) => (
    <PlaceholderRow key={row} index={row} rowRef={row === 0 ? tailRef : undefined} />
  ));
}

/** A new order starts at its first row, wherever the previous order had been scrolled to. */
function ScrollToTop({ order }: { order: RunOrder }) {
  const { scroller } = useInspectorTable();
  useEffect(() => {
    scroller?.scrollTo({ top: 0 });
  }, [scroller, order.key, order.descending]);
  return null;
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

/** Devtool-dense runs list: one row per agent run, in the server's order. */
export function AgentTracesTable({
  traces,
  isLoading,
  error,
  hasMore,
  isFetching = false,
  isPlaceholder = false,
  order = NEWEST,
  onOrderChange,
  onRetry,
  onLoadMore,
  rangeEmpty = false,
  onSetUpTracing,
  picks,
}: AgentTracesTableProps) {
  const settled = !isLoading && !error;
  const isEmpty = settled && !hasMore && traces.length === 0;
  const autoContinue = settled && hasMore && traces.length > 0 && !isPlaceholder;
  const { columnVisibility, onColumnVisibilityChange } = usePersistedColumnVisibility("lens-traces");
  const sorting = useMemo(() => toSorting(order), [order]);
  const columns = useMemo(() => (picks ? [pickColumn(picks), ...RUN_COLUMNS] : RUN_COLUMNS), [picks]);
  const tableOptions: TableOptions<TraceSummary> = {
    data: traces,
    columns,
    getRowId: runKey,
    autoResetAll: false,
    manualSorting: true,
    enableSorting: onOrderChange !== undefined,
    enableMultiSort: false,
    enableSortingRemoval: false,
    sortDescFirst: true,
    state: { columnVisibility, sorting },
    onColumnVisibilityChange,
    onSortingChange: (updater) => onOrderChange?.(fromSorting(updater, order)),
    getCoreRowModel: getCoreRowModel(),
  };
  const table = useReactTable(tableOptions);
  return (
    <InspectorTable.Root table={table} data-testid="runs-table">
      <ScrollToTop order={order} />
      <InspectorTable.Grid aria-label="Agent runs" aria-busy={isFetching} className="min-w-[900px] text-xs">
        <InspectorTable.Header />
        <InspectorTable.Body<TraceSummary>
          rowHeight={() => ROW_HEIGHT}
          className={cn("transition-[filter]", isPlaceholder && "blur-[1.5px]")}
          after={
            <>
              {isLoading && SKELETON_ROWS.map((row) => <PlaceholderRow key={row} index={row} />)}
              {autoContinue && <LoadMoreRows isFetching={isFetching} onLoadMore={onLoadMore} />}
            </>
          }
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
      {isLoading && (
        <p role="status" className="sr-only">
          Loading runs…
        </p>
      )}
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
      {isEmpty && <EmptyRuns rangeEmpty={rangeEmpty} onSetUpTracing={onSetUpTracing} />}
    </InspectorTable.Root>
  );
}
