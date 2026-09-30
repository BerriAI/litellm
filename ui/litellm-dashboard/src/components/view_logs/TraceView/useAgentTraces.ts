import { useInfiniteQuery } from "@tanstack/react-query";
import moment from "moment";
import { useMemo } from "react";

import { ApiError } from "@/lib/http/client";

import { agentTraceListCall } from "../../networking";
import { LIVE_TAIL_INTERVAL_MS } from "../log_filter_logic";
import type { TracePage, TraceSummary } from "./traceTypes";

const TRACING_NOT_ENABLED_STATUS = 501;
const TRACING_ROUTE_MISSING_STATUS = 404;

const isTracingUnavailable = (error: unknown): error is ApiError =>
  error instanceof ApiError && (error.status === TRACING_NOT_ENABLED_STATUS || error.status === TRACING_ROUTE_MISSING_STATUS);

interface UseAgentTracesOptions {
  accessToken: string;
  startTime: string;
  endTime: string;
  isCustomDate: boolean;
  isLiveTail: boolean;
  enabled: boolean;
}

export interface AgentTracesResult {
  traces: TraceSummary[];
  isLoading: boolean;
  isFetching: boolean;
  notEnabledDetail: string | null;
  error: Error | null;
  hasMore: boolean;
  loadMore: () => void;
  refetch: () => void;
}

/**
 * GET /v1/traces for the Logs page time range, cursor-paginated ("Load more").
 * Preset ranges re-read "now" on every fetch so live tail keeps moving the end bound.
 */
export function useAgentTraces({
  accessToken,
  startTime,
  endTime,
  isCustomDate,
  isLiveTail,
  enabled,
}: UseAgentTracesOptions): AgentTracesResult {
  const query = useInfiniteQuery<TracePage, Error>({
    queryKey: ["agentTraces", accessToken, startTime, endTime, isCustomDate],
    queryFn: ({ pageParam }) =>
      agentTraceListCall({
        accessToken,
        startMs: moment(startTime).valueOf(),
        endMs: isCustomDate ? moment(endTime).valueOf() : Date.now(),
        cursor: pageParam as string | null,
      }),
    initialPageParam: null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    enabled,
    retry: (failureCount, error) => !isTracingUnavailable(error) && failureCount < 1,
    refetchInterval: (q) => (isLiveTail && !isTracingUnavailable(q.state.error) ? LIVE_TAIL_INTERVAL_MS : false),
    refetchIntervalInBackground: false,
  });

  const traces = useMemo(() => query.data?.pages.flatMap((page) => page.data) ?? [], [query.data]);
  const notEnabled = isTracingUnavailable(query.error);
  const unavailableDetail =
    query.error instanceof ApiError && query.error.status === TRACING_ROUTE_MISSING_STATUS
      ? "This proxy does not expose /v1/traces. Update the proxy to enable agent tracing."
      : query.error?.message || "Agent tracing is not enabled";

  return {
    traces,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    notEnabledDetail: notEnabled ? unavailableDetail : null,
    error: notEnabled ? null : query.error,
    hasMore: query.hasNextPage,
    loadMore: () => void query.fetchNextPage(),
    refetch: () => void query.refetch(),
  };
}
