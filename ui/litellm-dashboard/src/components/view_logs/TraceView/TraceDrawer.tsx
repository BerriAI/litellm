"use client";

import { useQuery } from "@tanstack/react-query";
import { Copy } from "lucide-react";
import { useCallback, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { Checkbox } from "@/components/ui/checkbox";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { copyToClipboard } from "@/utils/dataUtils";

import { agentTraceCall } from "../../networking";
import { AgentGraph } from "./AgentGraph";
import { SpanDetail } from "./SpanDetail";
import { SpanTree } from "./SpanTree";
import { AgentTracePill, SpanStatusBadge } from "./TracePills";
import { TraceSteps } from "./TraceSteps";
import type { Trace } from "./traceTypes";
import {
  buildVisibleTree,
  firstErrorSpan,
  flattenTree,
  fmtCost,
  fmtMs,
  fmtTok,
  revealSpan,
  sortStepsByCost,
  spanRowIds,
  stepsFromSpans,
  subtreeStats,
  totalCacheRead,
  traceHasErrors,
  type TreeUiState,
} from "./traceUtils";
import { stepSelection, useTraceNavigation } from "./useTraceNavigation";

export type TraceViewMode = "steps" | "tree" | "graph";

interface TraceDrawerProps {
  open: boolean;
  traceId: string | null;
  /** Span to select on open (e.g. a child LLM row was clicked in the table). */
  initialSpanId?: string | null;
  accessToken: string;
  onClose: () => void;
  onOpenRequestLog?: (requestId: string) => void;
}

const DRAWER_WIDTH = "min(1180px, 80vw)";
const MODES: { id: TraceViewMode; label: string }[] = [
  { id: "steps", label: "Steps" },
  { id: "tree", label: "Tree" },
  { id: "graph", label: "Graph" },
];
const EMPTY_UI: TreeUiState = { collapsed: new Set(), groupShown: {} };

interface InitialView {
  mode: TraceViewMode;
  selectedId: string | null;
}

/** Errors open on the first failed span in the Tree; otherwise the Steps narrative. */
export function initialTraceView(trace: Trace, initialSpanId?: string | null): InitialView {
  if (initialSpanId && trace.spans.some((s) => s.span_id === initialSpanId)) {
    return { mode: "steps", selectedId: initialSpanId };
  }
  if (traceHasErrors(trace)) {
    const failed = firstErrorSpan(trace.spans);
    if (failed) return { mode: "tree", selectedId: failed.span_id };
  }
  const firstStep = stepsFromSpans(trace.spans)[0]?.span ?? trace.spans[0];
  return { mode: "steps", selectedId: firstStep?.span_id ?? null };
}

/** Tree state with the initially selected span (e.g. the first failure) scrolled into reach. */
function initialTreeUi(trace: Trace, selectedId: string | null): TreeUiState {
  if (!selectedId) return EMPTY_UI;
  const { children } = buildVisibleTree(trace.spans, false);
  return revealSpan(children, subtreeStats(trace.spans), EMPTY_UI, selectedId);
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-background px-4 py-2.5">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-0.5 text-[15px] font-semibold">
        {value} {sub && <small className="text-[11px] font-normal text-muted-foreground">{sub}</small>}
      </div>
    </div>
  );
}

export function TraceStats({ trace }: { trace: Trace }) {
  const { summary } = trace;
  return (
    <div aria-label="Trace stats" className="grid grid-cols-3 gap-px border-b bg-border sm:grid-cols-4 lg:grid-cols-7">
      <Stat label="Duration" value={fmtMs(summary.duration_ms)} />
      <Stat label="Agents" value={String(summary.agent_count)} />
      <Stat label="LLM calls" value={String(summary.llm_calls)} sub={summary.models.join(", ")} />
      <Stat label="Tool calls" value={String(summary.tool_calls)} />
      <Stat
        label="Tokens"
        value={fmtTok(summary.input_tokens + summary.output_tokens)}
        sub={`${fmtTok(summary.input_tokens)} in · ${fmtTok(summary.output_tokens)} out`}
      />
      <Stat label="Cache read" value={fmtTok(totalCacheRead(trace.spans))} sub="tokens" />
      <Stat label="Cost" value={fmtCost(summary.spend)} sub="via spend logs" />
    </div>
  );
}

function AgentCostSplit({ trace }: { trace: Trace }) {
  if (trace.agents.length < 2) return null;
  const sorted = [...trace.agents].sort((a, b) => (b.spend ?? 0) - (a.spend ?? 0));
  return (
    <div aria-label="Cost by agent" className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
      <span>Cost by agent:</span>
      {sorted.map((agent) => (
        <span key={agent.name}>
          <b className="font-medium text-foreground">{agent.name}</b> {fmtCost(agent.spend)}
        </span>
      ))}
    </div>
  );
}

interface TraceBodyProps {
  trace: Trace;
  accessToken: string;
  initialSpanId?: string | null;
  onOpenRequestLog?: (requestId: string) => void;
}

function TraceBody({ trace, accessToken, initialSpanId, onOpenRequestLog }: TraceBodyProps) {
  const [initial] = useState(() => initialTraceView(trace, initialSpanId));
  const [mode, setMode] = useState<TraceViewMode>(initial.mode);
  const [selectedId, setSelectedId] = useState<string | null>(initial.selectedId);
  const [showFramework, setShowFramework] = useState(false);
  const [sortByCost, setSortByCost] = useState(false);
  const [ui, setUi] = useState<TreeUiState>(() => initialTreeUi(trace, initial.selectedId));

  const { children, visibleCount } = useMemo(
    () => buildVisibleTree(trace.spans, showFramework),
    [trace.spans, showFramework],
  );
  const stats = useMemo(() => subtreeStats(trace.spans), [trace.spans]);
  const rows = useMemo(() => flattenTree(children, stats, ui), [children, stats, ui]);
  const timeSteps = useMemo(() => stepsFromSpans(trace.spans), [trace.spans]);
  const steps = useMemo(() => (sortByCost ? sortStepsByCost(timeSteps) : timeSteps), [timeSteps, sortByCost]);
  const selectedSpan = useMemo(
    () => trace.spans.find((s) => s.span_id === selectedId) ?? null,
    [trace.spans, selectedId],
  );

  const order = useMemo(
    () => (mode === "steps" ? steps.map((s) => s.span.span_id) : spanRowIds(rows)),
    [mode, steps, rows],
  );
  const handleStep = useCallback(
    (delta: number) => setSelectedId((current) => stepSelection(order, current, delta)),
    [order],
  );
  useTraceNavigation(mode !== "graph", handleStep);

  const toggleCollapse = useCallback((spanId: string) => {
    setUi((prev) => {
      const collapsed = new Set(prev.collapsed);
      if (collapsed.has(spanId)) collapsed.delete(spanId);
      else collapsed.add(spanId);
      return { ...prev, collapsed };
    });
  }, []);
  const showGroup = useCallback((key: string, shown: number) => {
    setUi((prev) => ({ ...prev, groupShown: { ...prev.groupShown, [key]: shown } }));
  }, []);
  const openInvocation = useCallback(
    (spanId: string) => {
      setUi((prev) => revealSpan(children, stats, prev, spanId));
      setSelectedId(spanId);
      setMode("tree");
    },
    [children, stats],
  );
  const changeMode = useCallback(
    (next: TraceViewMode) => {
      if (next === "tree" && selectedId) setUi((prev) => revealSpan(children, stats, prev, selectedId));
      setMode(next);
    },
    [children, stats, selectedId],
  );

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 overflow-auto md:grid-cols-[minmax(360px,46%)_1fr] md:overflow-hidden">
      <div className="flex max-h-[55vh] min-h-0 flex-col border-b md:max-h-none md:border-r md:border-b-0">
        <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
          <ButtonGroup aria-label="Trace view">
            {MODES.map((m) => (
              <Button
                key={m.id}
                size="xs"
                variant={mode === m.id ? "secondary" : "outline"}
                aria-pressed={mode === m.id}
                onClick={() => changeMode(m.id)}
              >
                {m.label}
              </Button>
            ))}
          </ButtonGroup>
          {mode === "tree" && (
            <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
              <Checkbox checked={showFramework} onCheckedChange={(checked) => setShowFramework(checked === true)} />
              Show framework spans
            </label>
          )}
          {mode === "steps" && (
            <label className="flex cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
              <Checkbox checked={sortByCost} onCheckedChange={(checked) => setSortByCost(checked === true)} />
              Sort by cost
            </label>
          )}
          <span className="ml-auto text-xs text-muted-foreground">
            {mode === "tree" ? `${visibleCount} of ${trace.spans.length} spans` : `${steps.length} steps`}
          </span>
        </div>
        {mode === "steps" && <TraceSteps steps={steps} selectedId={selectedId} onSelect={setSelectedId} />}
        {mode === "tree" && (
          <SpanTree
            rows={rows}
            traceDurationMs={trace.summary.duration_ms}
            selectedId={selectedId}
            onSelect={setSelectedId}
            onToggleCollapse={toggleCollapse}
            onGroupShow={showGroup}
          />
        )}
        {mode === "graph" && <AgentGraph agents={trace.agents} spans={trace.spans} onOpenInvocation={openInvocation} />}
      </div>
      {selectedSpan ? (
        <SpanDetail
          key={selectedSpan.span_id}
          accessToken={accessToken}
          traceId={trace.summary.trace_id}
          span={selectedSpan}
          onOpenRequestLog={onOpenRequestLog}
        />
      ) : (
        <div className="p-5 text-sm text-muted-foreground">Select a span to see its details.</div>
      )}
    </div>
  );
}

function TraceHeader({ trace, onClose }: { trace: Trace; onClose: () => void }) {
  const { summary } = trace;
  return (
    <div className="border-b px-5 pt-3.5 pb-3">
      <div className="flex flex-wrap items-center gap-2.5">
        <AgentTracePill label="Agent trace" />
        <SheetTitle className="text-base font-semibold">{summary.name}</SheetTitle>
        <span className="flex-1" />
        <span className="hidden text-xs text-muted-foreground sm:inline">
          <kbd className="rounded border px-1 font-mono">J</kbd> /{" "}
          <kbd className="rounded border px-1 font-mono">K</kbd> to move
        </span>
        <Button variant="outline" size="xs" onClick={onClose}>
          Close <kbd className="font-mono text-muted-foreground">Esc</kbd>
        </Button>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <SpanStatusBadge status={traceHasErrors(trace) ? "error" : "ok"} />
        <span>
          service <b className="font-mono font-medium text-foreground">{summary.service}</b>
        </span>
        <span>·</span>
        <span className="font-mono">trace {summary.trace_id}</span>
        <Button
          variant="ghost"
          size="icon-xs"
          aria-label="Copy trace id"
          onClick={() => void copyToClipboard(summary.trace_id)}
        >
          <Copy />
        </Button>
        <span>·</span>
        <span>{new Date(summary.start_time).toLocaleString()}</span>
      </div>
      <div className="mt-1.5">
        <AgentCostSplit trace={trace} />
      </div>
    </div>
  );
}

/** Right-side drawer for one agent trace: header, stats strip, Steps / Tree / Graph views and span details. */
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
        className="gap-0 overflow-hidden p-0 data-[side=right]:w-screen data-[side=right]:sm:max-w-none md:data-[side=right]:w-(--trace-drawer-width)"
        style={{ "--trace-drawer-width": DRAWER_WIDTH } as React.CSSProperties}
      >
        {traceQuery.isLoading && (
          <div role="status" aria-label="Loading trace" className="flex h-full items-center justify-center">
            <SheetTitle className="sr-only">Loading trace</SheetTitle>
            <UiLoadingSpinner className="size-8 text-primary" />
          </div>
        )}
        {traceQuery.isError && (
          <div className="p-6 text-sm text-destructive">
            <SheetTitle className="mb-2">Could not load trace</SheetTitle>
            {traceQuery.error.message}
          </div>
        )}
        {trace && (
          <div className="flex h-full min-h-0 flex-col">
            <TraceHeader trace={trace} onClose={onClose} />
            <TraceStats trace={trace} />
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
