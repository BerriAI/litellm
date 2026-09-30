"use client";

import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";

import type { TraceSummary } from "./traceTypes";
import { fmtCost, fmtMs, fmtRelative, stepCount, traceTitle } from "./traceUtils";

interface AgentTracesTableProps {
  traces: TraceSummary[];
  isLoading: boolean;
  error: Error | null;
  hasMore: boolean;
  onLoadMore: () => void;
  onOpenTrace: (traceId: string) => void;
  /** True when a search is narrowing the list (changes the empty message). */
  filtered?: boolean;
}

const NUM = "text-right tabular-nums";

const COLUMNS: { label: string; className?: string }[] = [
  { label: "Date", className: "w-24" },
  { label: "Service", className: "w-40" },
  { label: "Run" },
  { label: "Steps", className: cn("w-16", NUM) },
  { label: "Failed", className: cn("w-16", NUM) },
  { label: "Duration", className: cn("w-24", NUM) },
  { label: "Cost", className: cn("w-24", NUM) },
];

function RunRow({ trace, onOpen }: { trace: TraceSummary; onOpen: () => void }) {
  const title = traceTitle(trace);
  return (
    <TableRow
      data-testid="agent-trace-row"
      onClick={onOpen}
      className="h-9 cursor-pointer border-border/60 text-[13px] text-muted-foreground hover:bg-muted/40 [&:hover_.run-title]:underline"
    >
      <TableCell className="tabular-nums" title={new Date(trace.start_time).toLocaleString()}>
        {fmtRelative(trace.start_time)}
      </TableCell>
      <TableCell className="truncate">{trace.service}</TableCell>
      <TableCell>
        <div className="flex min-w-0 items-baseline gap-2">
          <span className="run-title truncate text-foreground decoration-border underline-offset-4" title={title}>
            {title}
          </span>
          {title !== trace.name && <span className="shrink-0 text-xs">{trace.name}</span>}
        </div>
      </TableCell>
      <TableCell className={NUM}>{stepCount(trace).toLocaleString()}</TableCell>
      <TableCell className={NUM}>{trace.error_count > 0 ? trace.error_count.toLocaleString() : ""}</TableCell>
      <TableCell className={NUM}>{fmtMs(trace.duration_ms)}</TableCell>
      <TableCell className={NUM}>{fmtCost(trace.spend)}</TableCell>
    </TableRow>
  );
}

/** Agent runs, one per row, Datadog-trace-list style: neutral, dense, numbers right-aligned. */
export function AgentTracesTable({
  traces,
  isLoading,
  error,
  hasMore,
  onLoadMore,
  onOpenTrace,
  filtered = false,
}: AgentTracesTableProps) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto rounded-md border border-border/60">
      <Table aria-label="Agent runs" className="table-fixed">
        <TableHeader>
          <TableRow className="border-border/60 hover:bg-transparent">
            {COLUMNS.map((column) => (
              <TableHead
                key={column.label}
                className={cn("h-8 text-xs font-medium text-muted-foreground", column.className)}
              >
                {column.label}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {traces.map((trace) => (
            <RunRow key={trace.trace_id} trace={trace} onOpen={() => onOpenTrace(trace.trace_id)} />
          ))}
        </TableBody>
      </Table>
      {isLoading && <div className="py-10 text-center text-[13px] text-muted-foreground">Loading runs…</div>}
      {error && (
        <div className="py-10 text-center text-[13px] text-muted-foreground">Could not load runs: {error.message}</div>
      )}
      {!isLoading && !error && traces.length === 0 && (
        <div className="py-10 text-center text-[13px] text-muted-foreground">
          {filtered ? "No runs match this search." : "No agent runs in this time range."}
        </div>
      )}
      {hasMore && (
        <div className="flex justify-center border-t border-border/60 py-1.5">
          <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={onLoadMore}>
            Load more
          </Button>
        </div>
      )}
    </div>
  );
}
