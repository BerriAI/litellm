import { type RunSelection, useTracesApi } from "../api";
import { keepPreviousData, useInfiniteQuery, useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { useMemo } from "react";

import { ApiError } from "@/lib/http/client";
import {
  isLive,
  LIVE_TAIL_INTERVAL_MS,
  type RelativeRange,
  type TimeWindow,
  timeWindow,
} from "@/components/shared/timeRange/timeRange";

import type { TracePage, TraceSummary } from "../types";
import type { RunOrder } from "./runOrder";

interface PagePosition {
  readonly window: TracePage["window"];
  readonly cursor: string;
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
  /** The search the server applies before paging. */
  q: string;
  /** A window inside the range that replaces it for the list. */
  zoom: TimeWindow | null;
  order: RunOrder;
}

export interface AgentTracesResult {
  traces: TraceSummary[];
  resolvedWindow: TracePage["window"] | null;
  isLoading: boolean;
  isFetching: boolean;
  /** Rows belong to the previous order, search or window while this one loads. */
  isPlaceholder: boolean;
  /** Set when the proxy answered 501: tracing isn't configured. */
  notEnabledDetail: string | null;
  error: Error | null;
  hasMore: boolean;
  loadMore: () => void;
  refetch: () => void;
}

/**
 * GET /v1/traces for the range (or the zoom inside it) and search, cursor-paginated as the runs list scrolls.
 * A live range rolls on refresh; subsequent pages keep the first page's window.
 */
export function useAgentTraces({
  accessToken,
  range,
  enabled,
  q,
  zoom,
  order,
}: UseAgentTracesOptions): AgentTracesResult {
  const traces = useTracesApi(accessToken);
  const isLiveTail = isLive(range);
  const fetchPage = async (pageParam: unknown): Promise<TracePage> => {
    const position = pageParam as PagePosition | null;
    const window = position
      ? { startMs: position.window.start_ms, endMs: position.window.end_ms }
      : zoom ?? timeWindow(range, Date.now());
    const selection: RunSelection = { window, q, ...(position ? { asOfMs: position.window.as_of_ms } : {}) };
    const page = { cursor: position?.cursor ?? null };
    return traces.list({ selection, order, page });
  };
  const queryOptions: Parameters<typeof useInfiniteQuery<TracePage, Error>>[0] = {
    queryKey: ["agentTraces", traces.scope, range.hours, range.anchorMs, q, zoom, order.key, order.descending],
    placeholderData: keepPreviousData,
    queryFn: ({ pageParam }) => fetchPage(pageParam),
    initialPageParam: null,
    getNextPageParam: (lastPage): PagePosition | undefined =>
      lastPage.next_cursor ? { window: lastPage.window, cursor: lastPage.next_cursor } : undefined,
    enabled,
    staleTime: LIVE_TAIL_INTERVAL_MS,
    retry: (failureCount, error) => !requiresUserAction(error) && failureCount < 1,
    refetchInterval: (q) => (isLiveTail && !requiresUserAction(q.state.error) ? LIVE_TAIL_INTERVAL_MS : false),
    refetchOnWindowFocus: (q) => !requiresUserAction(q.state.error),
    refetchOnReconnect: (q) => !requiresUserAction(q.state.error),
    refetchIntervalInBackground: false,
  };
  const query = useInfiniteQuery<TracePage, Error>(queryOptions);

  const loaded = useMemo(() => query.data?.pages.flatMap((page) => page.data) ?? [], [query.data]);
  const notEnabled = isTracingNotEnabled(query.error);

  return {
    traces: loaded,
    resolvedWindow: query.isPlaceholderData ? null : query.data?.pages[0]?.window ?? null,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    isPlaceholder: query.isPlaceholderData,
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
