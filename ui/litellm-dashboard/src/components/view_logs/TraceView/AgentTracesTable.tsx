"use client";

import { useQuery } from "@tanstack/react-query";
import { Workflow } from "lucide-react";
import { Fragment, useState } from "react";

import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatCellDate } from "@/components/shared/table_cells/date_cell";
import { cn } from "@/lib/cva.config";

import { agentTraceCall } from "../../networking";
import { AgentTracePill, SpanStatusBadge, SpanTypePill } from "./TracePills";
import type { Span, TraceSummary } from "./traceTypes";
import {
  agentBadgeLabel,
  fmtCost,
  fmtMs,
  fmtTok,
  llmSpans,
  previewText,
  shortId,
  spanLabel,
  summaryHasErrors,
  traceDisplayName,
} from "./traceUtils";

interface AgentTracesTableProps {
  accessToken: string;
  traces: TraceSummary[];
  isLoading: boolean;
  error: Error | null;
  hasMore: boolean;
  onLoadMore: () => void;
  /** Open the trace drawer, optionally focused on one span. */
  onOpenTrace: (traceId: string, spanId?: string) => void;
  /** Compact mode is used in "All" where the table sits above the request logs. */
  compact?: boolean;
}

const COLUMNS = ["", "Time", "Type", "Status", "Trace", "Input", "Cost", "Duration", "Model", "Tokens"] as const;

const TRUNC = "block max-w-[280px] truncate";

function ChildRows({
  accessToken,
  traceId,
  onOpenTrace,
}: {
  accessToken: string;
  traceId: string;
  onOpenTrace: AgentTracesTableProps["onOpenTrace"];
}) {
  const traceQuery = useQuery({
    queryKey: ["agentTrace", traceId, accessToken],
    queryFn: () => agentTraceCall(accessToken, traceId),
    staleTime: 30_000,
  });
  if (traceQuery.isLoading || traceQuery.isError) {
    return (
      <TableRow>
        <TableCell colSpan={COLUMNS.length} className="bg-muted/40 pl-10 text-xs text-muted-foreground">
          {traceQuery.isError ? `Could not load trace: ${traceQuery.error.message}` : "Loading LLM calls…"}
        </TableCell>
      </TableRow>
    );
  }
  const spans = llmSpans(traceQuery.data?.spans ?? []);
  return (
    <>
      {spans.map((span) => (
        <ChildRow key={span.span_id} span={span} onClick={() => onOpenTrace(traceId, span.span_id)} />
      ))}
    </>
  );
}

function ChildRow({ span, onClick }: { span: Span; onClick: () => void }) {
  const tokens = span.input_tokens + span.output_tokens;
  return (
    <TableRow
      onClick={onClick}
      className="cursor-pointer bg-muted/40 text-xs [&>td:first-child]:shadow-[inset_3px_0_0_var(--color-violet-200)]"
    >
      <TableCell />
      <TableCell className="text-muted-foreground">+{fmtMs(span.start_offset_ms)}</TableCell>
      <TableCell>
        <SpanTypePill type="llm" />
      </TableCell>
      <TableCell>
        <SpanStatusBadge status={span.status} />
      </TableCell>
      <TableCell className="font-mono text-[11px]">
        {span.litellm ? (
          shortId(span.litellm.request_id, 22)
        ) : (
          <span className="text-muted-foreground">{span.agent}</span>
        )}
      </TableCell>
      <TableCell className="text-muted-foreground">
        <span className={TRUNC}>{previewText(span.input_preview)}</span>
      </TableCell>
      <TableCell>{fmtCost(span.litellm?.spend)}</TableCell>
      <TableCell>{fmtMs(span.duration_ms)}</TableCell>
      <TableCell>{spanLabel(span)}</TableCell>
      <TableCell>{fmtTok(tokens)}</TableCell>
    </TableRow>
  );
}

function TraceRow({
  trace,
  expanded,
  onToggle,
  onOpen,
}: {
  trace: TraceSummary;
  expanded: boolean;
  onToggle: () => void;
  onOpen: () => void;
}) {
  const failed = summaryHasErrors(trace);
  // Roots that never closed report "unset"; with no errors the run reads as a success.
  const status = failed ? "error" : "ok";
  const handleToggle = (event: React.MouseEvent) => {
    event.stopPropagation();
    onToggle();
  };
  return (
    <TableRow
      data-testid="agent-trace-row"
      onClick={onOpen}
      className={cn("cursor-pointer", failed && "bg-destructive/5")}
    >
      <TableCell className="w-6">
        <button
          type="button"
          aria-label={expanded ? "Hide LLM calls" : "Show LLM calls"}
          aria-expanded={expanded}
          onClick={handleToggle}
          className={cn("inline-block w-3.5 text-muted-foreground transition-transform", expanded && "rotate-90")}
        >
          ▸
        </button>
      </TableCell>
      <TableCell className="text-muted-foreground">{formatCellDate(new Date(trace.start_time), "datetime")}</TableCell>
      <TableCell>
        <AgentTracePill label={agentBadgeLabel(trace)} />
      </TableCell>
      <TableCell>
        <SpanStatusBadge status={status} />
        {trace.error_count > 0 && <span className="ml-1 text-[11px] text-destructive">{trace.error_count} err</span>}
      </TableCell>
      <TableCell>
        <div className="font-medium">{traceDisplayName(trace)}</div>
        <div className="font-mono text-[11px] text-muted-foreground">{shortId(trace.trace_id)}</div>
      </TableCell>
      <TableCell>
        <span className={TRUNC} title={previewText(trace.input_preview)}>
          {previewText(trace.input_preview)}
        </span>
      </TableCell>
      <TableCell>{fmtCost(trace.spend)}</TableCell>
      <TableCell>{fmtMs(trace.duration_ms)}</TableCell>
      <TableCell>
        <span className={TRUNC}>{trace.models.join(", ")}</span>
      </TableCell>
      <TableCell>
        {fmtTok(trace.input_tokens + trace.output_tokens)}{" "}
        <span className="text-muted-foreground">
          ({fmtTok(trace.input_tokens)}+{fmtTok(trace.output_tokens)})
        </span>
      </TableCell>
    </TableRow>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Workflow className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium">No agent traces in this time range</div>
      <div className="max-w-xs text-center text-sm text-muted-foreground">
        Agent runs exported over OTLP to this proxy will appear here.
      </div>
    </div>
  );
}

/** Agent trace rows: violet badge, ▸ expands the trace's LLM calls, click opens the trace drawer. */
export function AgentTracesTable({
  accessToken,
  traces,
  isLoading,
  error,
  hasMore,
  onLoadMore,
  onOpenTrace,
  compact = false,
}: AgentTracesTableProps) {
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());
  const toggle = (traceId: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(traceId)) next.delete(traceId);
      else next.add(traceId);
      return next;
    });

  if (compact && traces.length === 0) return null;

  return (
    <div
      className={cn(
        "rounded-lg border",
        compact ? "mb-3 max-h-[40vh] shrink-0 overflow-y-auto" : "min-h-0 flex-1 overflow-y-auto",
      )}
    >
      <Table aria-label="Agent traces">
        <TableHeader>
          <TableRow>
            {COLUMNS.map((column, i) => (
              <TableHead key={i} className="text-xs text-muted-foreground">
                {column}
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {traces.map((trace) => (
            <Fragment key={trace.trace_id}>
              <TraceRow
                trace={trace}
                expanded={expanded.has(trace.trace_id)}
                onToggle={() => toggle(trace.trace_id)}
                onOpen={() => onOpenTrace(trace.trace_id)}
              />
              {expanded.has(trace.trace_id) && (
                <ChildRows accessToken={accessToken} traceId={trace.trace_id} onOpenTrace={onOpenTrace} />
              )}
            </Fragment>
          ))}
        </TableBody>
      </Table>
      {isLoading && <div className="py-6 text-center text-sm text-muted-foreground">Loading agent traces…</div>}
      {error && <div className="py-4 text-center text-sm text-destructive">Could not load traces: {error.message}</div>}
      {!isLoading && !error && traces.length === 0 && <EmptyState />}
      {hasMore && (
        <div className="flex justify-center border-t py-2">
          <Button variant="ghost" size="sm" onClick={onLoadMore}>
            Load more traces
          </Button>
        </div>
      )}
    </div>
  );
}
