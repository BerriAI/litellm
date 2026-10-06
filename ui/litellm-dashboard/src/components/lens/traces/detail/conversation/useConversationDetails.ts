import { useQueries, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo, useState } from "react";
import { useTracesApi } from "../../api";
import type { Span, SpanDetail, Trace } from "../../types";
import { CONVERSATION_PAGE_SIZE, conversationSteps } from "./conversation";

export function useConversationDetails(trace: Trace, accessToken: string) {
  const traces = useTracesApi(accessToken);
  const queryClient = useQueryClient();
  const [limit, setLimit] = useState(CONVERSATION_PAGE_SIZE);
  const steps = useMemo(() => conversationSteps(trace.spans), [trace.spans]);
  const visible = steps.slice(0, limit);
  const { trace_id: traceId, trace_ref: traceRef } = trace.summary;
  const spanQuery = useCallback(
    (span: Span) => ({
      queryKey: ["agentTraceSpan", traceId, traceRef, span.span_id, accessToken],
      queryFn: (): Promise<SpanDetail> => traces.span(traceId, span.span_id, traceRef),
      staleTime: Infinity,
      retry: false as const,
      retryOnMount: false,
    }),
    [traceId, traceRef, accessToken, traces],
  );
  const queries = useQueries({ queries: visible.map(spanQuery) });
  const unresolvedIndex = queries.findIndex((query) => !query.isSuccess);
  const loadedCount = unresolvedIndex < 0 ? queries.length : unresolvedIndex;
  const details = new Map(
    queries.slice(0, loadedCount).map((query, index) => [visible[index].span_id, query.data!] as const),
  );
  const loadMore = useCallback(
    async (signal: AbortSignal) => {
      const nextLimit = limit + CONVERSATION_PAGE_SIZE;
      const batch = steps.slice(limit, nextLimit);
      await Promise.all(batch.map((span) => queryClient.prefetchQuery(spanQuery(span))));
      if (!signal.aborted) setLimit(nextLimit);
    },
    [limit, steps, queryClient, spanQuery],
  );
  return {
    details,
    entries: visible.map((span, index) => ({ span, query: queries[index] })),
    complete: loadedCount === steps.length,
    loading: queries.some((query) => query.isPending),
    failed: queries.some((query) => query.isError),
    hasMore: visible.length < steps.length,
    loadMore,
  };
}
