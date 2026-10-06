import { useQueries, useQueryClient } from "@tanstack/react-query";
import { useCallback, useMemo, useState } from "react";
import { ApiError } from "@/lib/http/client";
import { useTracesApi, type TracesApi } from "../../api";
import type { Span, SpanDetail, Trace } from "../../types";
import { CONVERSATION_PAGE_SIZE, conversationSteps } from "./conversation";

interface BatchResult {
  details: SpanDetail[];
  failedIds: string[];
}

async function readBatch(traces: TracesApi, traceId: string, ids: string[], traceRef?: string): Promise<BatchResult> {
  try {
    const details = await traces.spans(traceId, ids, traceRef);
    const returned = new Set(details.map((detail) => detail.span_id));
    return { details, failedIds: ids.filter((id) => !returned.has(id)) };
  } catch (error) {
    if (!(error instanceof ApiError) || error.status !== 413 || ids.length <= 1) {
      return { details: [], failedIds: ids };
    }
    const middle = Math.ceil(ids.length / 2);
    const first = await readBatch(traces, traceId, ids.slice(0, middle), traceRef);
    const second = await readBatch(traces, traceId, ids.slice(middle), traceRef);
    return { details: [...first.details, ...second.details], failedIds: [...first.failedIds, ...second.failedIds] };
  }
}

export function useConversationDetails(trace: Trace, accessToken: string) {
  const traces = useTracesApi(accessToken);
  const queryClient = useQueryClient();
  const [limit, setLimit] = useState(CONVERSATION_PAGE_SIZE);
  const steps = useMemo(() => conversationSteps(trace.spans), [trace.spans]);
  const visible = steps.slice(0, limit);
  const { trace_id: traceId, trace_ref: traceRef, span_count: spanCount } = trace.summary;
  const cachedDetails = useCallback(
    (ids: string[]) =>
      ids.flatMap((id) => {
        const detail = queryClient.getQueryData<SpanDetail>(["agentTraceSpan", traceId, traceRef, id, accessToken]);
        return detail ? [detail] : [];
      }),
    [traceId, traceRef, accessToken, queryClient],
  );
  const batchQuery = useCallback(
    (batch: Span[]) => {
      const ids = batch.map((span) => span.span_id);
      const queryKey = ["agentTraceContents", traceId, traceRef, ids, spanCount, accessToken];
      return {
        queryKey,
        queryFn: async (): Promise<BatchResult> => {
          const previous = queryClient.getQueryData<BatchResult>(queryKey);
          const pending = previous?.failedIds.length ? previous.failedIds : ids;
          const result = await readBatch(traces, traceId, pending, traceRef);
          const details = [
            ...new Map(
              [...(previous?.details ?? []), ...cachedDetails(ids), ...result.details].map((detail) => [
                detail.span_id,
                detail,
              ]),
            ).values(),
          ];
          for (const detail of result.details) {
            queryClient.setQueryData(["agentTraceSpan", traceId, traceRef, detail.span_id, accessToken], detail);
          }
          return { details, failedIds: result.failedIds };
        },
        staleTime: 30_000,
        retry: false as const,
        retryOnMount: false,
      };
    },
    [traceId, traceRef, spanCount, accessToken, traces, queryClient, cachedDetails],
  );
  const batches = Array.from({ length: Math.ceil(visible.length / CONVERSATION_PAGE_SIZE) }, (_, index) =>
    visible.slice(index * CONVERSATION_PAGE_SIZE, (index + 1) * CONVERSATION_PAGE_SIZE),
  );
  const queries = useQueries({ queries: batches.map(batchQuery) });
  const complete = queries.every((query) => query.isSuccess && query.data.failedIds.length === 0);
  const details = new Map(
    [
      ...cachedDetails(visible.map((span) => span.span_id)),
      ...queries.flatMap((query) => query.data?.details ?? []),
    ].map((detail) => [detail.span_id, detail]),
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
    entries: batches.map((batch, index) => ({
      span: batch.find((span) => queries[index].data?.failedIds.includes(span.span_id)) ?? batch[0],
      failed: queries[index].isError || Boolean(queries[index].data?.failedIds.length),
      retry: queries[index].refetch,
    })),
    complete: complete && visible.length === steps.length,
    loading: queries.some((query) => query.isPending),
    failed: queries.some((query) => query.isError || Boolean(query.data?.failedIds.length)),
    hasMore: visible.length < steps.length,
    loadMore,
  };
}
