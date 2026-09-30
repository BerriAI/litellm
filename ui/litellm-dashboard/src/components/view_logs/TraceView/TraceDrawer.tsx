"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Check, Circle, Copy } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import { agentTraceCall, getProxyBaseUrl } from "../../networking";
import { DetailPane } from "./DetailPane";
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
export function initialRunSelection(trace: Trace): { selectedId: string; state: SpanTreeState } {
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
      className="shrink-0 gap-1.5 rounded-[4px] font-mono text-[10px] shadow-none"
      onClick={async () => setCopied(await copyToClipboard(agentHandoffText(traceId, null, traceRef), "Command copied"))}
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      {copied ? "Command copied" : "Copy for agent"}
    </Button>
  );
}

function Stat({ label, value, error = false }: { label: string; value: string; error?: boolean }) {
  return (
    <span className={cn("shrink-0", error && "text-destructive")}>
      <span className="text-muted-foreground/70">{label} </span>
      {value}
    </span>
  );
}

function RunHeader({ trace, onBack }: { trace: Trace; onBack: () => void }) {
  const { summary } = trace;
  const failed = summary.error_count > 0;
  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-border bg-card px-2">
      <button
        type="button"
        onClick={onBack}
        className="grid size-7 shrink-0 place-items-center rounded-[4px] text-muted-foreground hover:bg-muted hover:text-foreground"
        aria-label="Back to runs"
      >
        <ArrowLeft className="size-4" />
      </button>
      <Circle className={cn("size-2 shrink-0 fill-current", failed ? "text-destructive" : "text-muted-foreground")} />
      <h1 className="shrink-0 truncate text-[14px] font-medium text-foreground">{traceDisplayName(summary)}</h1>
      <span className="flex min-w-0 items-center gap-1 font-mono text-[11px] text-muted-foreground">
        <span className="truncate">{summary.trace_id}</span>
        <button
          type="button"
          aria-label="Copy trace ID"
          className="shrink-0 hover:text-foreground"
          onClick={() => void copyToClipboard(summary.trace_id, "Trace ID copied")}
        >
          <Copy className="size-3" />
        </button>
      </span>
      <div className="flex min-w-0 items-center gap-4 font-mono text-[11px] text-foreground tabular-nums">
        <Stat label="duration" value={fmtMs(summary.duration_ms)} />
        <Stat label="steps" value={summary.span_count.toLocaleString()} />
        {failed && <Stat label="failed" value={summary.error_count.toLocaleString()} error />}
      </div>
      <div className="ml-auto">
        <CopyForAgent traceId={summary.trace_id} traceRef={summary.trace_ref} />
      </div>
    </header>
  );
}

/** Tree + detail pane for one loaded run, with J/K/arrow keyboard navigation. */
function RunBody({ trace, accessToken }: { trace: Trace; accessToken: string }) {
  const initial = useMemo(() => initialRunSelection(trace), [trace]);
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
      if (["j", "J", "ArrowDown"].includes(event.key)) {
        event.preventDefault();
        const next = rows[Math.min(rows.length - 1, index + 1)];
        if (next) select(next.id);
      } else if (["k", "K", "ArrowUp"].includes(event.key)) {
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
  }, [rows, selectedRow, detailOpen, select, toggleSpan, toggleGroup]);

  return (
    <div
      className={cn(
        "grid min-h-0 flex-1",
        detailOpen
          ? "grid-cols-1 grid-rows-2 lg:grid-cols-[minmax(0,3fr)_minmax(360px,2fr)] lg:grid-rows-1 2xl:grid-cols-[minmax(0,2fr)_minmax(420px,1fr)]"
          : "grid-cols-1",
      )}
    >
      <SpanTree
        rows={rows}
        spanCount={trace.summary.span_count}
        totalMs={trace.summary.duration_ms}
        selectedId={selectedRow?.id ?? selectedId}
        hideFramework={state.hideFramework}
        onSelect={select}
        onToggleHideFramework={(hideFramework) => setState((prev) => ({ ...prev, hideFramework }))}
        onToggleSpan={toggleSpan}
        onToggleGroup={toggleGroup}
        onLoadMore={loadMore}
      />
      {detailOpen && (
        <DetailPane trace={trace} row={selectedRow} accessToken={accessToken} onClose={() => setDetailOpen(false)} />
      )}
    </div>
  );
}

interface RunViewProps {
  traceId: string;
  traceRef?: string;
  accessToken: string;
  onBack: () => void;
}

/** One agent run: header with totals and "Copy for agent", span tree on the left, span details on the right. */
export function RunView({ traceId, traceRef, accessToken, onBack }: RunViewProps) {
  const traceQuery = useQuery({
    queryKey: ["agentTrace", traceId, traceRef, accessToken],
    queryFn: () => agentTraceCall(accessToken, traceId, traceRef),
    staleTime: 30_000,
  });
  const trace = traceQuery.data;

  if (traceQuery.isLoading) {
    return (
      <div role="status" aria-label="Loading trace" className="flex h-[60vh] items-center justify-center">
        <UiLoadingSpinner className="size-6 text-muted-foreground" />
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
      className="flex min-h-[560px] flex-1 flex-col overflow-hidden border-y border-border bg-background"
      data-testid="run-view"
    >
      <RunHeader trace={trace} onBack={onBack} />
      <RunBody key={trace.summary.trace_id} trace={trace} accessToken={accessToken} />
    </div>
  );
}
