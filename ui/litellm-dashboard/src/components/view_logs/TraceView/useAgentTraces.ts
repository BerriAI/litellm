import { useTracesApi } from "./tracesApi";
import { useInfiniteQuery, useQuery, type UseQueryOptions } from "@tanstack/react-query";
import moment from "moment";
import { useMemo } from "react";

import { ApiError } from "@/lib/http/client";

import { LIVE_TAIL_INTERVAL_MS } from "../log_filter_logic";
import type { TracePage, TraceSummary } from "./traceTypes";
import type { TraceWindow } from "./tracesApi";

interface LoadedTracePage extends TracePage {
  window: TraceWindow;
}

export const TRACING_NOT_ENABLED_STATUS = 501;
/** A proxy without the tracing routes at all answers 404; treat it like tracing being off. */
const TRACING_ROUTE_MISSING_STATUS = 404;

export const isTracingNotEnabled = (error: unknown): boolean =>
  error instanceof ApiError &&
  (error.status === TRACING_NOT_ENABLED_STATUS || error.status === TRACING_ROUTE_MISSING_STATUS);

const requiresUserAction = (error: unknown): boolean => {
  if (!(error instanceof ApiError)) return false;
  return isTracingNotEnabled(error) || error.status === 401 || error.status === 403;
};

const displayError = (error: Error | null): Error | null => {
  if (!(error instanceof ApiError)) return error;
  if (error.status === 401) return new Error("Your session is no longer valid. Sign out and sign in again.");
  if (error.status === 403) return new Error("Your account does not have access to these traces.");
  return error;
};

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
 * GET /v1/traces for the Logs page time range, cursor-paginated as the runs list scrolls.
 * Preset ranges roll on refresh; subsequent pages keep the first page's window.
 */
export function useAgentTraces({
  accessToken,
  startTime,
  endTime,
  isCustomDate,
  isLiveTail,
  enabled,
}: UseAgentTracesOptions): AgentTracesResult {
  const traces = useTracesApi(accessToken);
  const fetchPage = async (pageParam: unknown): Promise<LoadedTracePage> => {
    const nowMs = Date.now();
    const window = (pageParam as TraceWindow | null) ?? {
      startMs: traceWindowStartMs(startTime, endTime, isCustomDate, nowMs),
      endMs: isCustomDate ? moment(endTime).valueOf() : nowMs,
    };
    return { ...(await traces.list(window)), window };
  };
  const queryOptions: Parameters<typeof useInfiniteQuery<LoadedTracePage, Error>>[0] = {
    queryKey: ["agentTraces", accessToken, startTime, endTime, isCustomDate],
    queryFn: ({ pageParam }) => fetchPage(pageParam),
    initialPageParam: null,
    getNextPageParam: (lastPage) =>
      lastPage.next_cursor ? { ...lastPage.window, cursor: lastPage.next_cursor } : undefined,
    enabled,
    staleTime: LIVE_TAIL_INTERVAL_MS,
    retry: (failureCount, error) => !requiresUserAction(error) && failureCount < 1,
    refetchInterval: (q) => (isLiveTail && !requiresUserAction(q.state.error) ? LIVE_TAIL_INTERVAL_MS : false),
    refetchOnWindowFocus: (q) => !requiresUserAction(q.state.error),
    refetchOnReconnect: (q) => !requiresUserAction(q.state.error),
    refetchIntervalInBackground: false,
  };
  const query = useInfiniteQuery<LoadedTracePage, Error>(queryOptions);

  const loaded = useMemo(() => query.data?.pages.flatMap((page) => page.data) ?? [], [query.data]);
  const notEnabled = isTracingNotEnabled(query.error);

  return {
    traces: loaded,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    notEnabledDetail: notEnabled ? query.error?.message || "Agent tracing is not enabled" : null,
    error: notEnabled ? null : displayError(query.error),
    hasMore: query.hasNextPage,
    loadMore: () => {
      if (query.hasNextPage && !query.isFetching) void query.fetchNextPage({ cancelRefetch: false });
    },
    refetch: () => void query.refetch(),
  };
}

export function useTraceAvailability(accessToken: string, enabled: boolean) {
  const traces = useTracesApi(accessToken);
  const options: UseQueryOptions<boolean, Error> = {
    queryKey: ["trace-availability", accessToken],
    queryFn: () => traces.anyRecorded(),
    enabled,
    retry: false,
    refetchInterval: (query) =>
      query.state.data || requiresUserAction(query.state.error) ? false : LIVE_TAIL_INTERVAL_MS,
    refetchOnWindowFocus: (query) => !requiresUserAction(query.state.error),
    refetchOnReconnect: (query) => !requiresUserAction(query.state.error),
    refetchIntervalInBackground: false,
  };
  return useQuery(options);
}
