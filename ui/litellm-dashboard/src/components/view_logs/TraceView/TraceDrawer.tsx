"use client";
import { type TraceHandoff, useTracesApi } from "./tracesApi";

import { QueryErrorResetBoundary, useQueryClient, useSuspenseInfiniteQuery } from "@tanstack/react-query";
import { ArrowLeft, Check, Copy } from "lucide-react";
import { Suspense, useCallback, useDeferredValue, useEffect, useMemo, useState } from "react";
import { ErrorBoundary } from "react-error-boundary";
import { useEventListener, useTimeout } from "usehooks-ts";

import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import { DetailPane } from "./DetailPane";
import { IdChip } from "./IdChip";
import { formatCost } from "./AgentTracesTable";
import { SpanIcon } from "./SpanIcon";
import { SpanTree } from "./SpanTree";
import { TraceConversation } from "./TraceConversation";
import { FrameworkLogo, traceFramework } from "./TraceFramework";
import { type RunSelection, traceKey } from "./traceRouting";
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

function CopyForAgent({ handoff }: { handoff: TraceHandoff }) {
  const [copied, setCopied] = useState(false);
  useTimeout(() => setCopied(false), copied ? 1600 : null);
  return (
    <Button
      variant="outline"
      size="xs"
      className="h-7 shrink-0 gap-1.5 rounded-md text-xs shadow-none"
      onClick={async () => setCopied(await copyToClipboard(handoff.text, handoff.copied))}
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

function RunHeader({
  trace,
  handoff,
  onBack,
  embedded,
}: {
  trace: Trace;
  handoff: TraceHandoff;
  onBack: () => void;
  embedded: boolean;
}) {
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
          <CopyForAgent handoff={handoff} />
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

interface RunBodyProps {
  trace: Trace;
  accessToken: string;
  selection: RunSelection;
  embedded: boolean;
  stale: boolean;
}

function RunBody({ trace, accessToken, selection, embedded, stale }: RunBodyProps) {
  const spanKeys = embedded ? EMBEDDED_SPAN_KEYS : SPAN_KEYS;
  const { view, selectSpan, setView, stepQuery: query, setStepQuery: setQuery, errorsOnly, setErrorsOnly } = selection;
  const [initial] = useState(() => initialRunSelection(trace, selection.spanId ?? undefined));
  const [state, setState] = useState<SpanTreeState>(initial.state);
  const selectedId = selection.spanId ?? initial.selectedId;
  const [detailOpen, setDetailOpen] = useState(true);
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
      selectSpan(id);
      setDetailOpen(true);
      setState((prev) => revealSpanInState(trace.spans, prev, id));
    },
    [trace.spans, selectSpan],
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

  const setRowExpanded = (row: TreeRow, expand: boolean) => {
    if (row.kind === "span" && row.hasChildren && row.collapsed === expand) toggleSpan(row.id);
    if (row.kind === "group" && row.expanded !== expand) toggleGroup(row.id);
  };
  useEventListener(
    "keydown",
    (event) => {
      if (stale || view !== "steps" || ignoreStepKey(event)) return;
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
    },
    undefined,
    true,
  );

  if (view === "conversation")
    return (
      <TabsContent value="conversation" className="flex min-h-0 flex-1">
        <TraceConversation
          trace={trace}
          accessToken={accessToken}
          onOpenStep={(id) => {
            select(id);
            setView("steps");
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
        <div className="min-h-0 min-w-0">
          <DetailPane
            trace={trace}
            row={selectedRow}
            accessToken={accessToken}
            spanTab={selection.spanTab}
            onSpanTabChange={selection.setSpanTab}
            onClose={() => setDetailOpen(false)}
          />
        </div>
      )}
    </TabsContent>
  );
}

interface RunViewProps {
  traceId: string;
  traceRef?: string;
  selection: RunSelection;
  accessToken: string;
  onBack: () => void;
  /** Rendered inside the side drawer: the drawer owns closing and sizing. */
  embedded?: boolean;
}

function selectedSpanMissing(trace: Trace, spanId: string | null): boolean {
  return Boolean(spanId && !trace.spans.some((span) => span.span_id === spanId));
}

function RunLoading({ embedded }: { embedded: boolean }) {
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

function RunLoadError({ error, onBack, onRetry }: { error: unknown; onBack: () => void; onRetry: () => void }) {
  return (
    <div className="p-6 text-xs" data-testid="run-view-error">
      <button
        type="button"
        onClick={onBack}
        className="mb-4 inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-3.5" /> Back to traces
      </button>
      <h1 className="mb-2 text-sm font-medium">Could not load trace</h1>
      <span className="text-muted-foreground">{error instanceof Error ? error.message : "Unknown error"}</span>
      <Button variant="outline" size="sm" className="ml-3" onClick={onRetry}>
        Retry
      </Button>
    </div>
  );
}

/** Keeps the shown run on screen, inert, while the next one loads. */
export function RunView(props: RunViewProps) {
  const traceId = useDeferredValue(props.traceId);
  const traceRef = useDeferredValue(props.traceRef);
  const switching = traceId !== props.traceId || traceRef !== props.traceRef;
  const shownKey = traceKey({ traceId, traceRef });
  return (
    <QueryErrorResetBoundary>
      {({ reset }) => (
        <ErrorBoundary
          onReset={reset}
          resetKeys={[shownKey]}
          fallbackRender={({ error, resetErrorBoundary }) => (
            <RunLoadError error={error} onBack={props.onBack} onRetry={resetErrorBoundary} />
          )}
        >
          <Suspense fallback={<RunLoading embedded={props.embedded ?? false} />}>
            <LoadedRun key={shownKey} {...props} traceId={traceId} traceRef={traceRef} switching={switching} />
          </Suspense>
        </ErrorBoundary>
      )}
    </QueryErrorResetBoundary>
  );
}

function LoadedRun({
  traceId,
  traceRef,
  selection,
  accessToken,
  onBack,
  embedded = false,
  switching,
}: RunViewProps & { switching: boolean }) {
  const traces = useTracesApi(accessToken);
  const queryClient = useQueryClient();
  const queryKey = ["agentTrace", traceId, traceRef, accessToken];
  const traceQueryOptions = {
    queryKey,
    queryFn: ({ pageParam }: { pageParam: string | null }) => traces.trace(traceId, traceRef, pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage: Trace) => lastPage.next_cursor ?? undefined,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    refetchOnMount: false,
    retry: false,
  };
  const traceQuery = useSuspenseInfiniteQuery(traceQueryOptions);
  const refreshTrace = () => queryClient.resetQueries({ queryKey, exact: true });
  const trace = useMemo(() => {
    const [first, ...rest] = traceQuery.data.pages;
    return { ...first, spans: [first, ...rest].flatMap((page) => page.spans) };
  }, [traceQuery.data]);
  const seekingSpan = !switching && selectedSpanMissing(trace, selection.spanId);
  const { hasNextPage, isFetching, isError, fetchNextPage } = traceQuery;
  const canSeek = seekingSpan && hasNextPage;
  useEffect(() => {
    if (canSeek && !isFetching && !isError) void fetchNextPage();
  }, [canSeek, isFetching, isError, fetchNextPage]);
  const pageAction = isError ? "Retry" : "Load more steps";

  return (
    <Tabs
      value={selection.view}
      onValueChange={(value) => selection.setView(value as RunSelection["view"])}
      className={cn(
        "@container/trace flex flex-1 flex-col gap-0 overflow-hidden bg-background",
        embedded ? "min-h-0" : "min-h-[560px] border-y border-border",
        switching && "opacity-60 transition-opacity delay-150 duration-150 motion-reduce:transition-none",
      )}
      aria-busy={switching}
      inert={switching}
      data-testid="run-view"
    >
      <RunHeader
        trace={trace}
        handoff={traces.handoff(trace.summary.trace_id, null, trace.summary.trace_ref)}
        onBack={onBack}
        embedded={embedded}
      />
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
        key={seekingSpan ? "seeking" : "loaded"}
        trace={trace}
        accessToken={accessToken}
        selection={selection}
        embedded={embedded}
        stale={switching}
      />
    </Tabs>
  );
}
