import { useTracesApi } from "../api";
import { keepPreviousData, useInfiniteQuery, useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { useMemo } from "react";

import {
  isLive,
  LIVE_TAIL_INTERVAL_MS,
  type RelativeRange,
  type TimeWindow,
  timeWindow,
} from "@/components/shared/timeRange/timeRange";
import { ApiError } from "@/lib/http/client";

import type { TracePage, TraceSummary } from "../types";
import type { TraceWindow } from "../api";

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
  range: RelativeRange;
  enabled: boolean;
}

export interface AgentTracesResult {
  traces: TraceSummary[];
  isLoading: boolean;
  isFetching: boolean;
  /** The previous range's rows, still shown while the newly picked range loads. */
  isPlaceholder: boolean;
  /** The window the shown rows were fetched for; lags the picked range while a placeholder is shown. */
  window: TimeWindow | null;
  /** Set when the proxy answered 501: tracing isn't configured. */
  notEnabledDetail: string | null;
  error: Error | null;
  hasMore: boolean;
  loadMore: () => void;
  refetch: () => void;
}

/**
 * GET /v1/traces for the Logs page time range, cursor-paginated as the runs list scrolls.
 * Preset ranges roll on refresh; subsequent pages keep the first page's window.
 */
export function useAgentTraces({ accessToken, range, enabled }: UseAgentTracesOptions): AgentTracesResult {
  const traces = useTracesApi(accessToken);
  const fetchPage = async (pageParam: unknown): Promise<LoadedTracePage> => {
    const window = (pageParam as TraceWindow | null) ?? timeWindow(range, Date.now());
    return { ...(await traces.list(window)), window };
  };
  const queryOptions: Parameters<typeof useInfiniteQuery<LoadedTracePage, Error>>[0] = {
    queryKey: ["agentTraces", accessToken, range.hours, range.anchorMs],
    queryFn: ({ pageParam }) => fetchPage(pageParam),
    initialPageParam: null,
    getNextPageParam: (lastPage) =>
      lastPage.next_cursor ? { ...lastPage.window, cursor: lastPage.next_cursor } : undefined,
    enabled,
    placeholderData: keepPreviousData,
    staleTime: LIVE_TAIL_INTERVAL_MS,
    retry: (failureCount, error) => !requiresUserAction(error) && failureCount < 1,
    refetchInterval: (q) => (isLive(range) && !requiresUserAction(q.state.error) ? LIVE_TAIL_INTERVAL_MS : false),
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
    isPlaceholder: query.isPlaceholderData,
    window: query.data?.pages[0]?.window ?? null,
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
