"use client";
import { useLensDemo } from "@/components/lens/LensDemoContext";
import { useTracesApi } from "@/components/lens/services";

import { useInfiniteQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Check, Copy } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import { getProxyBaseUrl } from "../../networking";
import { DetailPane } from "./DetailPane";
import { IdChip } from "./IdChip";
import { formatCost } from "./AgentTracesTable";
import { SpanIcon } from "./SpanIcon";
import { SpanTree } from "./SpanTree";
import { TraceConversation } from "./TraceConversation";
import { FrameworkLogo, traceFramework } from "./TraceFramework";
import type { SpanTreeState, TreeRow } from "./traceTree";
import type { Trace } from "./traceTypes";
import {
  buildTreeRows,
  firstErrorSpan,
  fmtMs,
  fmtTok,
  findTraceSteps,
  GROUP_PAGE_SIZE,
  isFrameworkSpan,
  nearestVisibleSpanId,
  revealSpanInState,
  traceAgentNames,
  traceDisplayName,
} from "./traceUtils";
import { ignoresLetterShortcut } from "../letterShortcut";

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
  const initialState = {
    ...INITIAL_STATE,
    collapsedSpanIds: new Set(
      trace.spans.filter((span) => span.type === "agent" && span.parent_span_id !== null).map((span) => span.span_id),
    ),
  };
  const failed = firstErrorSpan(trace.spans);
  if (!failed || failed.parent_span_id === null) {
    const root = trace.spans.find((s) => s.parent_span_id === null);
    return { selectedId: root?.span_id ?? "", state: initialState };
  }
  const visibleFailure = trace.spans
    .filter((s) => s.status === "error" && s.parent_span_id !== null && !isFrameworkSpan(s))
    .sort((a, b) => a.start_offset_ms - b.start_offset_ms)[0];
  const selectedId = visibleFailure?.span_id ?? nearestVisibleSpanId(trace.spans, failed.span_id, true);
  return { selectedId, state: revealSpanInState(trace.spans, initialState, selectedId) };
}

const toggle = (set: ReadonlySet<string>, id: string): Set<string> => {
  const next = new Set(set);
  if (next.has(id)) next.delete(id);
  else next.add(id);
  return next;
};

function CopyForAgent({ traceId, traceRef }: { traceId: string; traceRef?: string }) {
  const demo = useLensDemo();
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
        setCopied(
          await copyToClipboard(
            demo ? demo.copyTrace(traceId) : agentHandoffText(traceId, null, traceRef),
            demo ? "Trace copied" : "Command copied",
          ),
        )
      }
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      {copied ? "Copied" : "Copy for agent"}
    </Button>
  );
}

function Stat({ label, value, error = false }: { label: string; value: string; error?: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 text-xs tabular-nums",
        error ? "text-destructive" : "text-muted-foreground",
      )}
    >
      <span>{label}</span> <span className={cn("font-medium", !error && "text-foreground")}>{value}</span>
    </span>
  );
}

function RunIcon({ summary, failed }: { summary: Trace["summary"]; failed: boolean }) {
  const framework = traceFramework(summary);
  if (!framework) return <SpanIcon type="agent" error={failed} size="lg" />;
  return (
    <span
      className="inline-flex shrink-0 items-center gap-1.5 text-sm text-foreground"
      data-testid="run-framework"
      title={framework.label}
    >
      <FrameworkLogo framework={framework} />
      {traceAgentNames(summary).join(", ") || framework.label}
    </span>
  );
}

function RunHeader({ trace, onBack, embedded }: { trace: Trace; onBack: () => void; embedded: boolean }) {
  const { summary } = trace;
  const failed = summary.status === "error";
  return (
    <header className="shrink-0 border-b bg-background px-4 py-3">
      <div className="flex min-w-0 items-center gap-2">
        {!embedded && (
          <Button variant="ghost" size="icon-xs" onClick={onBack} aria-label="Back to runs">
            <ArrowLeft className="size-4" />
          </Button>
        )}
        <RunIcon summary={summary} failed={failed} />
        <h1 className="min-w-0 truncate text-base font-semibold">{traceDisplayName(summary)}</h1>
        <IdChip value={summary.trace_id} label="Copy trace ID" />
        <TabsList aria-label="Trace view" className="ml-auto shrink-0 group-data-horizontal/tabs:h-8">
          <TabsTrigger value="steps" className="text-xs">
            Steps
          </TabsTrigger>
          <TabsTrigger value="conversation" className="text-xs">
            Conversation
          </TabsTrigger>
        </TabsList>
        <div className="shrink-0">
          <CopyForAgent traceId={summary.trace_id} traceRef={summary.trace_ref} />
        </div>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2">
        <span className={cn("text-xs font-medium", failed ? "text-destructive" : "text-muted-foreground")}>
          {failed ? "Failed" : "Completed"}
        </span>
        <Stat label="Duration" value={fmtMs(summary.duration_ms)} />
        <Stat label="Steps" value={summary.span_count.toLocaleString()} />
        <Stat label="Tokens" value={fmtTok(summary.input_tokens + summary.output_tokens)} />
        <Stat label="Cost" value={summary.spend == null ? "Not reported" : formatCost(summary.spend)} />
        {summary.error_count > 0 && <Stat label="Step errors" value={summary.error_count.toLocaleString()} error />}
      </div>
    </header>
  );
}

/** Tree + detail pane for one loaded run, with J/K/arrow keyboard navigation. */
const SPAN_KEYS = { down: ["j", "J", "ArrowDown"], up: ["k", "K", "ArrowUp"] } as const;
const EMBEDDED_SPAN_KEYS = { down: ["ArrowDown"], up: ["ArrowUp"] } as const;

function ignoreStepKey(event: KeyboardEvent): boolean {
  const control = (event.target as HTMLElement | null)?.closest(
    "input, textarea, select, [contenteditable='true'], [role='combobox'], [role='tablist'], [role='menu'], [role='separator']",
  );
  return event.defaultPrevented || ignoresLetterShortcut(event) || Boolean(control);
}

type TraceView = "steps" | "conversation";

interface RunBodyProps {
  trace: Trace;
  accessToken: string;
  initialSpanId?: string;
  embedded: boolean;
  view: TraceView;
  onViewChange: (view: TraceView) => void;
}

function RunBody({ trace, accessToken, initialSpanId, embedded, view, onViewChange }: RunBodyProps) {
  const spanKeys = embedded ? EMBEDDED_SPAN_KEYS : SPAN_KEYS;
  const initial = useMemo(() => initialRunSelection(trace, initialSpanId), [trace, initialSpanId]);
  const [state, setState] = useState<SpanTreeState>(initial.state);
  const [selectedId, setSelectedId] = useState<string>(initial.selectedId);
  const [detailOpen, setDetailOpen] = useState(true);
  const [query, setQuery] = useState("");
  const [errorsOnly, setErrorsOnly] = useState(false);
  const filtering = Boolean(query.trim()) || errorsOnly;

  const treeRows = useMemo(() => buildTreeRows(trace.spans, state), [trace, state]);
  const rows = useMemo<TreeRow[]>(
    () =>
      filtering
        ? findTraceSteps(trace.spans, query, errorsOnly, state.hideFramework).map((span) => ({
            kind: "span",
            id: span.span_id,
            span,
            depth: 0,
            hasChildren: false,
            collapsed: false,
          }))
        : treeRows,
    [trace, state.hideFramework, query, errorsOnly, filtering, treeRows],
  );
  const selectedRow: TreeRow | undefined =
    rows.find((row) => row.id === selectedId) ?? treeRows.find((row) => row.id === selectedId) ?? rows[0];

  const select = useCallback(
    (id: string) => {
      setSelectedId(id);
      setDetailOpen(true);
      setState((prev) => revealSpanInState(trace.spans, prev, id));
    },
    [trace.spans],
  );
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
    if (view !== "steps") return;
    const setRowExpanded = (row: TreeRow, expand: boolean) => {
      if (row.kind === "span" && row.hasChildren && row.collapsed === expand) toggleSpan(row.id);
      if (row.kind === "group" && row.expanded !== expand) toggleGroup(row.id);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (ignoreStepKey(event)) return;
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
        setRowExpanded(row, false);
      } else if (event.key === "ArrowRight" && row) {
        setRowExpanded(row, true);
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, [rows, selectedRow, detailOpen, select, toggleSpan, toggleGroup, spanKeys, view]);

  if (view === "conversation")
    return (
      <TabsContent value="conversation" className="flex min-h-0 flex-1">
        <TraceConversation
          trace={trace}
          accessToken={accessToken}
          onOpenStep={(id) => {
            select(id);
            onViewChange("steps");
          }}
        />
      </TabsContent>
    );

  return (
    <TabsContent
      value="steps"
      className={cn(
        "grid min-h-0 flex-1",
        detailOpen
          ? "grid-cols-1 grid-rows-2 @[640px]/trace:grid-cols-[clamp(280px,38%,340px)_minmax(0,1fr)] @[640px]/trace:grid-rows-1"
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
        onOpenDetails={detailOpen ? undefined : () => setDetailOpen(true)}
        embedded={embedded}
        query={query}
        onQueryChange={setQuery}
        errorsOnly={errorsOnly}
        onErrorsOnlyChange={setErrorsOnly}
        filtering={filtering}
        onClearFilters={() => {
          setQuery("");
          setErrorsOnly(false);
        }}
        onCollapseAll={() =>
          setState((prev) => ({
            ...prev,
            collapsedSpanIds: new Set(
              trace.spans.filter((span) => span.parent_span_id !== null).map((span) => span.span_id),
            ),
            expandedGroupIds: new Set(),
          }))
        }
      />
      {detailOpen && (
        <div className="min-h-0 min-w-0 animate-slide-left motion-reduce:animate-none">
          <DetailPane trace={trace} row={selectedRow} accessToken={accessToken} onClose={() => setDetailOpen(false)} />
        </div>
      )}
    </TabsContent>
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

function initialSpanMissing(trace: Trace | undefined, spanId?: string): boolean {
  return Boolean(spanId && trace && !trace.spans.some((span) => span.span_id === spanId));
}

export function RunView({ traceId, traceRef, initialSpanId, accessToken, onBack, embedded = false }: RunViewProps) {
  const traces = useTracesApi(accessToken);
  const queryClient = useQueryClient();
  const [view, setView] = useState<TraceView>("steps");
  const traceQueryOptions = {
    queryKey: ["agentTrace", traceId, traceRef, accessToken],
    queryFn: ({ pageParam }: { pageParam: string | null }) => traces.trace(traceId, traceRef, pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage: Trace) => lastPage.next_cursor ?? undefined,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false,
    retry: false,
  };
  const traceQuery = useInfiniteQuery(traceQueryOptions);
  const refreshTrace = () => queryClient.resetQueries({ queryKey: traceQueryOptions.queryKey, exact: true });
  const trace = useMemo(() => {
    const pages = traceQuery.data?.pages;
    if (!pages?.length) return undefined;
    return { ...pages[0], spans: pages.flatMap((page) => page.spans) };
  }, [traceQuery.data]);
  const seekingSpan = initialSpanMissing(trace, initialSpanId);
  const { hasNextPage, isFetching, isError, fetchNextPage } = traceQuery;
  const canSeek = seekingSpan && hasNextPage;
  useEffect(() => {
    if (canSeek && !isFetching && !isError) void fetchNextPage();
  }, [canSeek, isFetching, isError, fetchNextPage]);
  const pageAction = isError ? "Retry" : "Load more steps";

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
  if (!trace) {
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
        <Button variant="outline" size="sm" className="ml-3" onClick={() => void refreshTrace()}>
          Retry
        </Button>
      </div>
    );
  }
  return (
    <Tabs
      value={view}
      onValueChange={(value) => setView(value as TraceView)}
      className={cn(
        "@container/trace flex flex-1 flex-col gap-0 overflow-hidden bg-background",
        embedded ? "min-h-0 animate-view-fade-in motion-reduce:animate-none" : "min-h-[560px] border-y border-border",
      )}
      data-testid="run-view"
    >
      <RunHeader trace={trace} onBack={onBack} embedded={embedded} />
      {(traceQuery.hasNextPage || traceQuery.isError) && (
        <div className="flex items-center justify-between gap-3 border-b px-3 py-2 text-xs" role="status">
          <span>
            {traceQuery.isError
              ? "Could not load more steps. Your loaded steps are still available."
              : `Showing ${trace.spans.length.toLocaleString()} of ${trace.summary.span_count.toLocaleString()} steps`}
          </span>
          {traceQuery.isError && (
            <Button size="xs" variant="ghost" disabled={traceQuery.isFetching} onClick={() => void refreshTrace()}>
              Refresh trace
            </Button>
          )}
          <Button
            size="xs"
            variant="outline"
            disabled={traceQuery.isFetching}
            onClick={() => void (traceQuery.hasNextPage ? traceQuery.fetchNextPage() : refreshTrace())}
          >
            {traceQuery.isFetching ? "Loading…" : pageAction}
          </Button>
        </div>
      )}
      <RunBody
        key={`${trace.summary.trace_ref || trace.summary.trace_id}:${initialSpanId && !seekingSpan ? initialSpanId : "root"}`}
        trace={trace}
        accessToken={accessToken}
        initialSpanId={initialSpanId}
        embedded={embedded}
        view={view}
        onViewChange={setView}
      />
    </Tabs>
  );
}
