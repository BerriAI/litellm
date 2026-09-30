import { useInfiniteQuery } from "@tanstack/react-query";
import moment from "moment";
import { useMemo } from "react";

import { ApiError } from "@/lib/http/client";

import { agentTraceListCall } from "../../networking";
import { LIVE_TAIL_INTERVAL_MS } from "../log_filter_logic";
import type { TracePage, TraceSummary } from "./traceTypes";

export const TRACING_NOT_ENABLED_STATUS = 501;

export const isTracingNotEnabled = (error: unknown): error is ApiError =>
  error instanceof ApiError && error.status === TRACING_NOT_ENABLED_STATUS;

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
  /** Set when the proxy answered 501: tracing isn't configured. */
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
    retry: (failureCount, error) => !isTracingNotEnabled(error) && failureCount < 1,
    refetchInterval: (q) => (isLiveTail && !isTracingNotEnabled(q.state.error) ? LIVE_TAIL_INTERVAL_MS : false),
    refetchIntervalInBackground: false,
  });

  const traces = useMemo(() => query.data?.pages.flatMap((page) => page.data) ?? [], [query.data]);
  const notEnabled = isTracingNotEnabled(query.error);

  return {
    traces,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    notEnabledDetail: notEnabled ? query.error?.message || "Agent tracing is not enabled" : null,
    error: notEnabled ? null : query.error,
    hasMore: query.hasNextPage,
    loadMore: () => void query.fetchNextPage(),
    refetch: () => void query.refetch(),
  };
}
