"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Check, Copy } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import { agentTraceCall, getProxyBaseUrl } from "../../networking";
import { DetailPane } from "./DetailPane";
import { IdChip } from "./IdChip";
import { formatCost } from "./AgentTracesTable";
import { SpanIcon } from "./SpanIcon";
import { SpanTree } from "./SpanTree";
import type { SpanTreeState, TreeRow } from "./traceTree";
import type { Trace } from "./traceTypes";
import {
  buildTreeRows,
  firstErrorSpan,
  fmtMs,
  GROUP_PAGE_SIZE,
  isFrameworkSpan,
  nearestVisibleSpanId,
  revealSpanInState,
  traceDisplayName,
} from "./traceUtils";

/** What "Copy for agent" puts on the clipboard: a one-liner Claude Code / Codex can run. */
export const agentHandoffText = (traceId: string, spanId?: string | null, traceRef?: string): string => {
  const url = `${getProxyBaseUrl().replace(/\/$/, "")}/v1/traces/${traceId}?format=md${spanId ? `&span_id=${spanId}` : ""}${traceRef ? `&trace_ref=${traceRef}` : ""}`;
  const what = spanId ? "this step of a LiteLLM agent trace" : "this LiteLLM agent trace";
  return `Read ${what} and explain what happened and why it failed:\ncurl -s -H "Authorization: Bearer $LITELLM_API_KEY" "${url}"`;
};

const INITIAL_STATE: SpanTreeState = {
  hideFramework: true,
  collapsedSpanIds: new Set(),
  expandedGroupIds: new Set(),
  groupRevealCounts: {},
};

/** First failed span if the run has errors (with its tree path opened), otherwise the root agent. */
export function initialRunSelection(
  trace: Trace,
  initialSpanId?: string,
): { selectedId: string; state: SpanTreeState } {
  if (initialSpanId && trace.spans.some((span) => span.span_id === initialSpanId)) {
    const selectedId = nearestVisibleSpanId(trace.spans, initialSpanId, false);
    const state = revealSpanInState(trace.spans, { ...INITIAL_STATE, hideFramework: false }, selectedId);
    return { selectedId, state };
  }
  const failed = firstErrorSpan(trace.spans);
  if (!failed || failed.parent_span_id === null) {
    const root = trace.spans.find((s) => s.parent_span_id === null);
    return { selectedId: root?.span_id ?? "", state: INITIAL_STATE };
  }
  const visibleFailure = trace.spans
    .filter((s) => s.status === "error" && s.parent_span_id !== null && !isFrameworkSpan(s))
    .sort((a, b) => a.start_offset_ms - b.start_offset_ms)[0];
  const selectedId = visibleFailure?.span_id ?? nearestVisibleSpanId(trace.spans, failed.span_id, true);
  return { selectedId, state: revealSpanInState(trace.spans, INITIAL_STATE, selectedId) };
}

const toggle = (set: ReadonlySet<string>, id: string): Set<string> => {
  const next = new Set(set);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
};

function CopyForAgent({ traceId, traceRef }: { traceId: string; traceRef?: string }) {
  const [copied, setCopied] = useState(false);
  useEffect(() => {
    if (!copied) return;
    const timeout = window.setTimeout(() => setCopied(false), 1600);
    return () => window.clearTimeout(timeout);
  }, [copied]);
  return (
    <Button
      variant="outline"
      size="xs"
      className="h-7 shrink-0 gap-1.5 rounded-md text-[12px] shadow-none"
      onClick={async () =>
        setCopied(await copyToClipboard(agentHandoffText(traceId, null, traceRef), "Command copied"))
      }
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      {copied ? "Command copied" : "Copy for agent"}
    </Button>
  );
}

function Stat({ label, value, error = false }: { label: string; value: string; error?: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-[5px] border border-border bg-card px-1.5 py-px text-[12px] tabular-nums",
        error && "border-destructive/40 bg-destructive/10 text-destructive",
      )}
    >
      <span className={cn("text-muted-foreground", error && "text-destructive/80")}>{label} </span>
      {value}
    </span>
  );
}

function RunHeader({ trace, onBack, embedded }: { trace: Trace; onBack: () => void; embedded: boolean }) {
  const { summary } = trace;
  const failed = summary.error_count > 0;
  return (
    <header className="flex min-h-11 shrink-0 flex-wrap items-center gap-x-2 gap-y-1.5 border-b border-border bg-card px-3 py-1.5">
      {!embedded && (
        <>
          <button
            type="button"
            onClick={onBack}
            className="grid size-7 shrink-0 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
            aria-label="Back to runs"
          >
            <ArrowLeft className="size-4" />
          </button>
          <span className="mx-1 h-[18px] w-px bg-border" />
        </>
      )}
      <SpanIcon type="agent" error={failed} size="lg" />
      <h1 className="min-w-0 truncate text-[14px] font-medium text-foreground">{traceDisplayName(summary)}</h1>
      <IdChip value={summary.trace_id} label="Copy trace ID" showValue />
      <div className="flex min-w-0 flex-wrap items-center gap-1.5">
        <Stat label="duration" value={fmtMs(summary.duration_ms)} />
        <Stat label="steps" value={summary.span_count.toLocaleString()} />
        <Stat label="cost" value={summary.spend == null ? "—" : formatCost(summary.spend)} />
        {failed && <Stat label="failed" value={summary.error_count.toLocaleString()} error />}
      </div>
      <div className="ml-auto">
        <CopyForAgent traceId={summary.trace_id} traceRef={summary.trace_ref} />
      </div>
    </header>
  );
}

/** Tree + detail pane for one loaded run, with J/K/arrow keyboard navigation. */
const SPAN_KEYS = { down: ["j", "J", "ArrowDown"], up: ["k", "K", "ArrowUp"] } as const;
const EMBEDDED_SPAN_KEYS = { down: ["ArrowDown"], up: ["ArrowUp"] } as const;

interface RunBodyProps {
  trace: Trace;
  accessToken: string;
  initialSpanId?: string;
  embedded: boolean;
}

function RunBody({ trace, accessToken, initialSpanId, embedded }: RunBodyProps) {
  const spanKeys = embedded ? EMBEDDED_SPAN_KEYS : SPAN_KEYS;
  const initial = useMemo(() => initialRunSelection(trace, initialSpanId), [trace, initialSpanId]);
  const [state, setState] = useState<SpanTreeState>(initial.state);
  const [selectedId, setSelectedId] = useState<string>(initial.selectedId);
  const [detailOpen, setDetailOpen] = useState(true);

  const rows = useMemo(() => buildTreeRows(trace.spans, state), [trace, state]);
  const selectedRow: TreeRow | undefined = rows.find((row) => row.id === selectedId) ?? rows[0];

  const select = useCallback((id: string) => {
    setSelectedId(id);
    setDetailOpen(true);
  }, []);
  const toggleSpan = useCallback(
    (id: string) => setState((prev) => ({ ...prev, collapsedSpanIds: toggle(prev.collapsedSpanIds, id) })),
    [],
  );
  const toggleGroup = useCallback(
    (id: string) => setState((prev) => ({ ...prev, expandedGroupIds: toggle(prev.expandedGroupIds, id) })),
    [],
  );
  const loadMore = useCallback(
    (groupId: string) =>
      setState((prev) => ({
        ...prev,
        groupRevealCounts: {
          ...prev.groupRevealCounts,
          [groupId]: (prev.groupRevealCounts[groupId] ?? GROUP_PAGE_SIZE) + GROUP_PAGE_SIZE,
        },
      })),
    [],
  );

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.target as HTMLElement | null)?.matches("input, textarea, [role='combobox']")) return;
      const index = rows.findIndex((row) => row.id === selectedRow?.id);
      const row = rows[index];
      if (event.key === "Escape" && detailOpen) {
        event.preventDefault();
        event.stopPropagation();
        setDetailOpen(false);
        return;
      }
      if ((spanKeys.down as readonly string[]).includes(event.key)) {
        event.preventDefault();
        const next = rows[Math.min(rows.length - 1, index + 1)];
        if (next) select(next.id);
      } else if ((spanKeys.up as readonly string[]).includes(event.key)) {
        event.preventDefault();
        const next = rows[Math.max(0, index - 1)];
        if (next) select(next.id);
      } else if (event.key === "ArrowLeft" && row) {
        if (row.kind === "span" && row.hasChildren && !row.collapsed) toggleSpan(row.id);
        if (row.kind === "group" && row.expanded) toggleGroup(row.id);
      } else if (event.key === "ArrowRight" && row) {
        if (row.kind === "span" && row.hasChildren && row.collapsed) toggleSpan(row.id);
        if (row.kind === "group" && !row.expanded) toggleGroup(row.id);
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [rows, selectedRow, detailOpen, select, toggleSpan, toggleGroup, spanKeys]);

  return (
    <div
      className={cn(
        "grid min-h-0 flex-1",
        detailOpen
          ? "grid-cols-1 grid-rows-2 lg:grid-cols-[minmax(340px,400px)_minmax(0,1fr)] lg:grid-rows-1"
          : "grid-cols-1",
      )}
    >
      <SpanTree
        rows={rows}
        summary={trace.summary}
        selectedId={selectedRow?.id ?? selectedId}
        hideFramework={state.hideFramework}
        onSelect={select}
        onToggleHideFramework={(hideFramework) => setState((prev) => ({ ...prev, hideFramework }))}
        onToggleSpan={toggleSpan}
        onToggleGroup={toggleGroup}
        onLoadMore={loadMore}
        embedded={embedded}
      />
      {detailOpen && (
        <div className="min-h-0 min-w-0 animate-slide-left motion-reduce:animate-none">
          <DetailPane trace={trace} row={selectedRow} accessToken={accessToken} onClose={() => setDetailOpen(false)} />
        </div>
      )}
    </div>
  );
}

interface RunViewProps {
  traceId: string;
  traceRef?: string;
  initialSpanId?: string;
  accessToken: string;
  onBack: () => void;
  /** Rendered inside the side drawer: the drawer owns closing and sizing. */
  embedded?: boolean;
}

/** One agent run: header with totals and "Copy for agent", span tree on the left, span details on the right. */
export function RunView({ traceId, traceRef, initialSpanId, accessToken, onBack, embedded = false }: RunViewProps) {
  const traceQuery = useQuery({
    queryKey: ["agentTrace", traceId, traceRef, accessToken],
    queryFn: () => agentTraceCall(accessToken, traceId, traceRef),
    staleTime: 30_000,
  });
  const trace = traceQuery.data;

  if (traceQuery.isLoading) {
    return (
      <div
        role="status"
        aria-label="Loading trace"
        className={embedded ? "flex flex-col gap-3 p-4" : "flex h-[60vh] items-center justify-center"}
      >
        {embedded ? (
          [72, 48, 88, 60, 80].map((w) => (
            <div key={w} className="h-4 animate-pulse rounded bg-trace-row-hover" style={{ width: `${w}%` }} />
          ))
        ) : (
          <UiLoadingSpinner className="size-6 text-muted-foreground" />
        )}
      </div>
    );
  }
  if (traceQuery.isError || !trace) {
    return (
      <div className="p-6 text-[12px]" data-testid="run-view-error">
        <button
          type="button"
          onClick={onBack}
          className="mb-4 inline-flex items-center gap-1.5 text-[12px] text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-3.5" /> Back to traces
        </button>
        <h1 className="mb-2 text-[13px] font-medium">Could not load trace</h1>
        <span className="text-muted-foreground">{traceQuery.error?.message ?? "Unknown error"}</span>
      </div>
    );
  }
  return (
    <div
      className={cn(
        "flex flex-1 flex-col overflow-hidden bg-background",
        embedded ? "min-h-0 animate-view-fade-in motion-reduce:animate-none" : "min-h-[560px] border-y border-border",
      )}
      data-testid="run-view"
    >
      <RunHeader trace={trace} onBack={onBack} embedded={embedded} />
      <RunBody
        key={trace.summary.trace_id}
        trace={trace}
        accessToken={accessToken}
        initialSpanId={initialSpanId}
        embedded={embedded}
      />
    </div>
  );
}
