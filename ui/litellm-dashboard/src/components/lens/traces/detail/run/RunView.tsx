"use client";

import { QueryErrorResetBoundary, useQueryClient, useSuspenseInfiniteQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import { Suspense, useDeferredValue, useEffect, useMemo } from "react";
import { ErrorBoundary } from "react-error-boundary";

import { Button } from "@/components/ui/button";
import { Tabs } from "@/components/ui/tabs";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";

import { useTracesApi } from "../../api";
import { classifyTraceReadFailure, traceReadRetry, traceReadRetryDelay } from "../../list/traceReadFailure";
import { type RunSelection, traceKey } from "../../routing";
import type { Trace } from "../../types";
import { PagingBanner } from "./PagingBanner";
import { RunBody } from "./RunBody";
import { RunHeader } from "./RunHeader";

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
    retry: traceReadRetry,
    retryDelay: traceReadRetryDelay,
  };
  const traceQuery = useSuspenseInfiniteQuery(traceQueryOptions);
  const refreshTrace = () => queryClient.resetQueries({ queryKey, exact: true });
  const failure = traceQuery.error ? classifyTraceReadFailure(traceQuery.error) : null;
  const trace = useMemo(() => {
    const pages = traceQuery.data.pages;
    return {
      ...pages[0],
      spans: pages.flatMap((page) => page.spans),
      next_cursor: pages[pages.length - 1].next_cursor,
    };
  }, [traceQuery.data]);
  const seekingSpan = !switching && selectedSpanMissing(trace, selection.spanId);
  const { hasNextPage, isFetching, isError, fetchNextPage } = traceQuery;
  const canSeek = seekingSpan && hasNextPage;
  useEffect(() => {
    if (canSeek && !isFetching && !isError) void fetchNextPage();
  }, [canSeek, isFetching, isError, fetchNextPage]);

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
      {(traceQuery.hasNextPage || failure) && (
        <PagingBanner
          loaded={trace.spans.length}
          total={trace.summary.span_count}
          failure={failure}
          busy={traceQuery.isFetching}
          onLoadMore={() => void traceQuery.fetchNextPage()}
          onRefresh={() => void refreshTrace()}
        />
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
