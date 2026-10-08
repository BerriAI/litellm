"use client";

import { QueryErrorResetBoundary, useQueryClient, useSuspenseInfiniteQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { Suspense, useDeferredValue, useEffect, useMemo, useState } from "react";
import { ErrorBoundary } from "react-error-boundary";

import { LoadingState } from "@/components/shared/LoadingState";
import { Button } from "@/components/ui/button";
import { Tabs } from "@/components/ui/tabs";
import { cn } from "@/lib/cva.config";

import { useTracesApi } from "../../api";
import { classifyTraceReadFailure, traceReadRetry, traceReadRetryDelay } from "../../list/traceReadFailure";
import { type RunSelection, traceKey } from "../../routing";
import type { Trace } from "../../types";
import { PagingBanner } from "./PagingBanner";
import { RunBody } from "./RunBody";
import { RunHeader } from "./RunHeader";
import { useTraceSignalFlags } from "../../list/useTraceSignals";

interface RunViewProps {
  traceId: string;
  traceRef?: string;
  selection: RunSelection;
  accessToken: string;
  onBack: () => void;
  /** Rendered inside the side drawer: the drawer owns closing and sizing. */
  embedded?: boolean;
  showSignals?: boolean;
}

function selectedSpanMissing(trace: Trace, spanId: string | null): boolean {
  return Boolean(spanId && !trace.spans.some((span) => span.span_id === spanId));
}

function RunLoading() {
  return <LoadingState title="Loading trace…" description="Fetching this run and its steps." />;
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
          <Suspense fallback={<RunLoading />}>
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
  showSignals = false,
  switching,
}: RunViewProps & { switching: boolean }) {
  const traces = useTracesApi(accessToken);
  const queryClient = useQueryClient();
  const [live, setLive] = useState(traces.live);
  const [manualRead, setManualRead] = useState(false);
  const queryKey = ["agentTrace", traceId, traceRef, accessToken];
  const traceQueryOptions = {
    queryKey,
    queryFn: ({ pageParam }: { pageParam: string | null }) => traces.trace(traceId, traceRef, pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage: Trace) => lastPage.next_cursor ?? undefined,
    staleTime: 30_000,
    refetchOnWindowFocus: live,
    refetchOnReconnect: live,
    refetchOnMount: true,
    refetchInterval: live ? 30_000 : (false as const),
    retry: traceReadRetry,
    retryDelay: traceReadRetryDelay,
  };
  const traceQuery = useSuspenseInfiniteQuery(traceQueryOptions);
  const signals = useTraceSignalFlags(accessToken, { trace_id: traceId, trace_ref: traceRef }, showSignals);
  const refreshTrace = () => queryClient.resetQueries({ queryKey, exact: true });
  const failure = traceQuery.isFetchNextPageError ? classifyTraceReadFailure(traceQuery.error) : null;
  const readManually = (read: () => Promise<unknown>) => {
    if (manualRead) return;
    setManualRead(true);
    void read().finally(() => setManualRead(false));
  };
  const refreshRun = () =>
    readManually(async () => {
      const refreshed = await traceQuery.refetch();
      if (refreshed.isError) return;
      const contentRef = refreshed.data?.pages[0].summary.trace_ref ?? traceRef;
      await queryClient.invalidateQueries({
        queryKey: ["agentTraceSpan", traceId, contentRef],
        predicate: (query) => query.queryKey.at(-1) === accessToken,
      });
    });
  const toggleLive = () => {
    if (live && !manualRead && !traceQuery.isFetchingNextPage) {
      void queryClient.cancelQueries({ queryKey, exact: true });
    }
    setLive((enabled) => !enabled);
  };
  const trace = useMemo(() => {
    const pages = traceQuery.data.pages;
    return {
      ...pages[0],
      spans: pages.flatMap((page) => page.spans),
      next_cursor: pages[pages.length - 1].next_cursor,
    };
  }, [traceQuery.data]);
  const seekingSpan = !switching && selectedSpanMissing(trace, selection.spanId);
  const { hasNextPage, isFetching, isFetchNextPageError, fetchNextPage } = traceQuery;
  const canSeek = seekingSpan && hasNextPage;
  useEffect(() => {
    if (canSeek && !isFetching && !isFetchNextPageError) void fetchNextPage();
  }, [canSeek, isFetching, isFetchNextPageError, fetchNextPage]);

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
        refreshing={manualRead || traceQuery.isFetching}
        onRefresh={refreshRun}
        live={live}
        canLive={traces.live}
        onLiveChange={toggleLive}
        signals={signals}
      />
      {traceQuery.isRefetchError && (
        <div role="alert" className="flex items-center gap-3 border-b p-3 text-xs text-muted-foreground">
          Could not refresh this run. Previously received steps are still shown.
          <Button variant="outline" size="sm" disabled={manualRead || traceQuery.isFetching} onClick={refreshRun}>
            Retry refresh
          </Button>
        </div>
      )}
      {(traceQuery.hasNextPage || failure) && (
        <PagingBanner
          loaded={trace.spans.length}
          total={trace.summary.span_count}
          failure={failure}
          busy={manualRead || traceQuery.isFetching}
          onLoadMore={() => readManually(() => traceQuery.fetchNextPage())}
          onRefresh={() => readManually(refreshTrace)}
        />
      )}
      <RunBody
        key={seekingSpan ? "seeking" : "loaded"}
        trace={trace}
        accessToken={accessToken}
        selection={selection}
        embedded={embedded}
        stale={switching}
        conversationPaging={{
          loading: manualRead || traceQuery.isFetching,
          failed: isFetchNextPageError,
          loadMore: fetchNextPage,
        }}
      />
    </Tabs>
  );
}
