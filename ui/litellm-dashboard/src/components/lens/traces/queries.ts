import { infiniteQueryOptions, queryOptions } from "@tanstack/react-query";

import type { TracesApi } from "./api";
import { traceReadRetry, traceReadRetryDelay } from "./list/traceReadFailure";
import type { TraceRef } from "./routing";
import type { Trace } from "./types";

export const TRACE_DETAIL_STALE_MS = 30_000;

const refParam = (traceRef: string | undefined): string | undefined => traceRef || undefined;

export const traceKeys = {
  trace: (accessToken: string, ref: TraceRef) =>
    ["agentTrace", ref.traceId, refParam(ref.traceRef), accessToken] as const,
  spans: (traceId: string, traceRef: string | undefined) => ["agentTraceSpan", traceId, refParam(traceRef)] as const,
  span: (accessToken: string, traceId: string, traceRef: string | undefined, spanId: string) =>
    ["agentTraceSpan", traceId, refParam(traceRef), spanId, accessToken] as const,
};

export function traceDetailQuery(traces: TracesApi, accessToken: string, ref: TraceRef) {
  const options = {
    queryKey: traceKeys.trace(accessToken, ref),
    queryFn: ({ pageParam }: { pageParam: string | null }) =>
      traces.trace(ref.traceId, refParam(ref.traceRef), pageParam),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage: Trace) => lastPage.next_cursor ?? undefined,
    staleTime: TRACE_DETAIL_STALE_MS,
    retry: traceReadRetry,
    retryDelay: traceReadRetryDelay,
  };
  return infiniteQueryOptions(options);
}

interface SpanRef extends TraceRef {
  readonly spanId: string;
}

export function spanDetailQuery(traces: TracesApi, accessToken: string, { traceId, traceRef, spanId }: SpanRef) {
  const options = {
    queryKey: traceKeys.span(accessToken, traceId, traceRef, spanId),
    queryFn: () => traces.span(traceId, spanId, refParam(traceRef)),
    staleTime: Infinity,
  };
  return queryOptions(options);
}
