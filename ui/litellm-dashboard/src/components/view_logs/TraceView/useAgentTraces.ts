import { useInfiniteQuery } from "@tanstack/react-query";
import moment from "moment";
import { useMemo } from "react";

import { ApiError } from "@/lib/http/client";

import { agentTraceListCall } from "../../networking";
import { LIVE_TAIL_INTERVAL_MS } from "../log_filter_logic";
import type { TracePage, TraceSummary } from "./traceTypes";

export const TRACING_NOT_ENABLED_STATUS = 501;
/** A proxy without the tracing routes at all answers 404; treat it like tracing being off. */
const TRACING_ROUTE_MISSING_STATUS = 404;

export const isTracingNotEnabled = (error: unknown): error is ApiError =>
  error instanceof ApiError &&
  (error.status === TRACING_NOT_ENABLED_STATUS || error.status === TRACING_ROUTE_MISSING_STATUS);

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

/** Start of the fetch window: a preset range rolls with "now", so live tail keeps a fixed-length window. */
export const traceWindowStartMs = (startTime: string, endTime: string, isCustomDate: boolean, nowMs: number): number =>
  isCustomDate ? moment(startTime).valueOf() : nowMs - (moment(endTime).valueOf() - moment(startTime).valueOf());

/**
 * GET /v1/traces for the Logs page time range, cursor-paginated ("Load more").
 * Preset ranges re-read "now" on every fetch, moving both bounds so the window keeps its length.
 */
export function useAgentTraces({
  accessToken,
  startTime,
  endTime,
  isCustomDate,
  isLiveTail,
  enabled,
}: UseAgentTracesOptions): AgentTracesResult {
  const fetchPage = (pageParam: unknown): Promise<TracePage> => {
    const nowMs = Date.now();
    const listOptions: Parameters<typeof agentTraceListCall>[0] = {
      accessToken,
      startMs: traceWindowStartMs(startTime, endTime, isCustomDate, nowMs),
      endMs: isCustomDate ? moment(endTime).valueOf() : nowMs,
      cursor: pageParam as string | null,
    };
    return agentTraceListCall(listOptions);
  };
  const queryOptions: Parameters<typeof useInfiniteQuery<TracePage, Error>>[0] = {
    queryKey: ["agentTraces", accessToken, startTime, endTime, isCustomDate],
    queryFn: ({ pageParam }) => fetchPage(pageParam),
    initialPageParam: null,
    getNextPageParam: (lastPage) => lastPage.next_cursor ?? undefined,
    enabled,
    retry: (failureCount, error) => !isTracingNotEnabled(error) && failureCount < 1,
    refetchInterval: (q) => (isLiveTail && !isTracingNotEnabled(q.state.error) ? LIVE_TAIL_INTERVAL_MS : false),
    refetchIntervalInBackground: false,
  };
  const query = useInfiniteQuery<TracePage, Error>(queryOptions);

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
