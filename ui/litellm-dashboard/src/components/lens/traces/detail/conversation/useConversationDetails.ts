import { useQueries, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo, useState } from "react";
import { ApiError } from "@/lib/http/client";
import { useTracesApi, type TracesApi } from "../../api";
import type { Span, SpanDetail, Trace } from "../../types";
import { CONVERSATION_PAGE_SIZE, conversationSteps } from "./conversation";

async function readBatch(traces: TracesApi, traceId: string, ids: string[], traceRef: string): Promise<SpanDetail[]> {
  try {
    return await traces.spans(traceId, ids, traceRef);
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 413 || ids.length <= 1) throw error;
    const middle = Math.ceil(ids.length / 2);
    const first = await readBatch(traces, traceId, ids.slice(0, middle), traceRef);
    return [...first, ...(await readBatch(traces, traceId, ids.slice(middle), traceRef))];
  }
}

export function useConversationDetails(trace: Trace, accessToken: string) {
  const traces = useTracesApi(accessToken);
  const queryClient = useQueryClient();
  const [limit, setLimit] = useState(CONVERSATION_PAGE_SIZE);
  const steps = useMemo(() => conversationSteps(trace.spans), [trace.spans]);
  const visible = steps.slice(0, limit);
  const { trace_id: traceId, trace_ref: traceRef, span_count: spanCount } = trace.summary;
  const batchQuery = useCallback(
    (batch: Span[]) => {
      const ids = batch.map((span) => span.span_id);
      return {
        queryKey: ["agentTraceContents", traceId, traceRef, ids, spanCount, accessToken],
        queryFn: async (): Promise<SpanDetail[]> => {
          const details = await readBatch(traces, traceId, ids, traceRef);
          const returned = new Set(details.map((detail) => detail.span_id));
          if (ids.some((id) => !returned.has(id))) throw new Error("Some requested spans are unavailable");
          for (const detail of details) {
            queryClient.setQueryData(["agentTraceSpan", traceId, traceRef, detail.span_id, accessToken], detail);
          }
          return details;
        },
        staleTime: 30_000,
        retry: false as const,
        retryOnMount: false,
      };
    },
    [traceId, traceRef, spanCount, accessToken, traces, queryClient],
  );
  const batches = Array.from({ length: Math.ceil(visible.length / CONVERSATION_PAGE_SIZE) }, (_, index) =>
    visible.slice(index * CONVERSATION_PAGE_SIZE, (index + 1) * CONVERSATION_PAGE_SIZE),
  );
  const queries = useQueries({ queries: batches.map(batchQuery) });
  const unresolvedIndex = queries.findIndex((query) => !query.isSuccess);
  const loadedCount = unresolvedIndex < 0 ? queries.length : unresolvedIndex;
  const details = new Map(
    queries.slice(0, loadedCount).flatMap((query) => query.data!.map((detail) => [detail.span_id, detail] as const)),
  );
  const loadMore = useCallback(
    async (signal: AbortSignal) => {
      const nextLimit = limit + CONVERSATION_PAGE_SIZE;
      const batch = steps.slice(limit, nextLimit);
      await queryClient.prefetchQuery(batchQuery(batch));
      if (!signal.aborted) setLimit(nextLimit);
    },
    [limit, steps, queryClient, batchQuery],
  );
  return {
    details,
    entries: batches.map((batch, index) => ({ span: batch[0], query: queries[index] })),
    complete: loadedCount === batches.length && visible.length === steps.length,
    loading: queries.some((query) => query.isPending),
    failed: queries.some((query) => query.isError),
    hasMore: visible.length < steps.length,
    loadMore,
  };
}
