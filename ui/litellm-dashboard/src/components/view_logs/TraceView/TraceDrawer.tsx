"use client";

import { useQuery } from "@tanstack/react-query";
import { Copy, X } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { copyToClipboard } from "@/utils/dataUtils";

import { agentTraceCall, agentTraceMarkdownUrl } from "../../networking";
import { SpanDetail } from "./SpanDetail";
import { TraceOutline } from "./TraceOutline";
import type { Trace } from "./traceTypes";
import {
  buildOutline,
  copyForAgentText,
  fmtCost,
  fmtMs,
  fmtRelative,
  FOLD_PAGE_SIZE,
  initialOutlineSelection,
  isRowOpen,
  selectableIds,
  stepCount,
  traceHasErrors,
  traceTitle,
  visibleOutlineRows,
  type OutlineRow,
  type OutlineUiState,
} from "./traceUtils";
import { stepSelection, useTraceNavigation } from "./useTraceNavigation";

interface TraceDrawerProps {
  open: boolean;
  traceId: string | null;
  /** Span to select on open. */
  initialSpanId?: string | null;
  accessToken: string;
  onClose: () => void;
  onOpenRequestLog?: (requestId: string) => void;
}

const DRAWER_WIDTH = "min(1280px, 88vw)";

function Figure({ value, label }: { value: string; label: string }) {
  return (
    <div className="min-w-0">
      <div className="text-2xl font-semibold tracking-tight tabular-nums">{value}</div>
      <div className="mt-0.5 text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

/** Duration · Cost · Steps · Agents (· Failed): type and whitespace, no tiles. */
export function TraceFigures({ trace }: { trace: Trace }) {
  const { summary } = trace;
  return (
    <div aria-label="Trace stats" className="mt-5 flex flex-wrap gap-x-12 gap-y-4">
      <Figure value={fmtMs(summary.duration_ms)} label="Duration" />
      <Figure value={fmtCost(summary.spend)} label="Cost" />
      <Figure value={stepCount(summary).toLocaleString()} label="Steps" />
      <Figure value={summary.agent_count.toLocaleString()} label="Agents" />
      {summary.error_count > 0 && <Figure value={summary.error_count.toLocaleString()} label="Failed" />}
    </div>
  );
}

function TraceHeader({ trace, onClose }: { trace: Trace; onClose: () => void }) {
  const { summary } = trace;
  const copyForAgent = () =>
    void copyToClipboard(
      copyForAgentText(agentTraceMarkdownUrl(summary.trace_id), traceHasErrors(trace)),
      "Copied — paste into Claude or Codex",
    );
  return (
    <header className="border-b border-border/60 px-8 pt-6 pb-6">
      <div className="flex items-start gap-6">
        <div className="min-w-0 flex-1">
          <SheetTitle className="line-clamp-2 text-xl leading-snug font-semibold tracking-tight">
            {traceTitle(summary)}
          </SheetTitle>
          <div className="mt-1.5 text-[13px] text-muted-foreground">
            {summary.name} · {summary.service} · {fmtRelative(summary.start_time)}
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <Button variant="outline" size="sm" onClick={copyForAgent}>
            <Copy /> Copy for agent
          </Button>
          <Button variant="ghost" size="icon-sm" aria-label="Close" onClick={onClose}>
            <X />
          </Button>
        </div>
      </div>
      <TraceFigures trace={trace} />
    </header>
  );
}

interface TraceBodyProps {
  trace: Trace;
  accessToken: string;
  initialSpanId?: string | null;
  onOpenRequestLog?: (requestId: string) => void;
}

function TraceBody({ trace, accessToken, initialSpanId, onOpenRequestLog }: TraceBodyProps) {
  const allRows = useMemo(() => buildOutline(trace), [trace]);
  const [initial] = useState(() => initialOutlineSelection(allRows, initialSpanId));
  const [selectedId, setSelectedId] = useState<string>(initial.selectedId);
  const [ui, setUi] = useState<OutlineUiState>(initial.ui);

  const rows = useMemo(() => visibleOutlineRows(allRows, ui), [allRows, ui]);
  const order = useMemo(() => selectableIds(rows), [rows]);
  const selectedRow = useMemo(() => allRows.find((r) => r.id === selectedId) ?? null, [allRows, selectedId]);

  const handleStep = useCallback(
    (delta: number) => setSelectedId((current) => stepSelection(order, current, delta) ?? current),
    [order],
  );
  useTraceNavigation(true, handleStep);

  const isOpen = useCallback((row: OutlineRow) => isRowOpen(row, ui), [ui]);
  const toggle = useCallback(
    (row: OutlineRow) => setUi((prev) => ({ ...prev, open: { ...prev.open, [row.id]: !isRowOpen(row, prev) } })),
    [],
  );
  const select = useCallback(
    (row: OutlineRow) => {
      setSelectedId(row.id);
      // Folds and failure groups open when selected: their content is the rows below.
      if ((row.kind === "agents" || row.kind === "failures") && !isRowOpen(row, ui)) toggle(row);
    },
    [toggle, ui],
  );
  const showMore = useCallback((row: OutlineRow) => {
    if (!row.page) return;
    const fold = row.page.fold;
    setUi((prev) => ({ ...prev, shown: { ...prev.shown, [fold]: (prev.shown[fold] ?? FOLD_PAGE_SIZE) + FOLD_PAGE_SIZE } }));
  }, []);

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 overflow-auto md:grid-cols-[minmax(400px,46%)_1fr] md:overflow-hidden">
      <div className="flex max-h-[55vh] min-h-0 flex-col border-b border-border/60 md:max-h-none md:border-r md:border-b-0">
        <TraceOutline
          rows={rows}
          traceDurationMs={trace.summary.duration_ms}
          selectedId={selectedId}
          isOpen={isOpen}
          onSelect={select}
          onToggle={toggle}
          onShowMore={showMore}
        />
      </div>
      {selectedRow ? (
        <SpanDetail
          key={selectedRow.id}
          accessToken={accessToken}
          traceId={trace.summary.trace_id}
          row={selectedRow}
          onOpenRequestLog={onOpenRequestLog}
        />
      ) : (
        <div className="p-6 text-sm text-muted-foreground">Select a step.</div>
      )}
    </div>
  );
}

/** Full-height sheet for one agent run: header with big numbers, outline on the left, detail on the right. */
export function TraceDrawer({
  open,
  traceId,
  initialSpanId,
  accessToken,
  onClose,
  onOpenRequestLog,
}: TraceDrawerProps) {
  const traceQuery = useQuery({
    queryKey: ["agentTrace", traceId, accessToken],
    queryFn: () => agentTraceCall(accessToken, traceId as string),
    enabled: open && traceId !== null,
    staleTime: 30_000,
  });
  const trace = traceQuery.data;

  return (
    <Sheet open={open} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <SheetContent
        side="right"
        showCloseButton={false}
        className="gap-0 overflow-hidden bg-background p-0 data-[side=right]:w-screen data-[side=right]:sm:max-w-none md:data-[side=right]:w-(--trace-drawer-width)"
        style={{ "--trace-drawer-width": DRAWER_WIDTH } as React.CSSProperties}
      >
        {traceQuery.isLoading && (
          <div role="status" aria-label="Loading trace" className="flex h-full items-center justify-center">
            <SheetTitle className="sr-only">Loading trace</SheetTitle>
            <UiLoadingSpinner className="size-6 text-muted-foreground" />
          </div>
        )}
        {traceQuery.isError && (
          <div className="p-8 text-sm text-muted-foreground">
            <SheetTitle className="mb-2 text-foreground">Could not load this run</SheetTitle>
            {traceQuery.error.message}
          </div>
        )}
        {trace && (
          <div className="flex h-full min-h-0 flex-col">
            <TraceHeader trace={trace} onClose={onClose} />
            <TraceBody
              key={`${trace.summary.trace_id}:${initialSpanId ?? ""}`}
              trace={trace}
              accessToken={accessToken}
              initialSpanId={initialSpanId}
              onOpenRequestLog={onOpenRequestLog}
            />
          </div>
        )}
      </SheetContent>
    </Sheet>
  );
}
