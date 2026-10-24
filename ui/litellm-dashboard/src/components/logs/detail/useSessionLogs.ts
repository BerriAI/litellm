import { useInfiniteQuery } from "@tanstack/react-query";
import { useEffect, useMemo } from "react";

import { sessionSpendLogsCall } from "../../networking";
import { LogEntry } from "../types";

// page_size is the backend maximum (le=100). The endpoint returns newest-first,
// so the page cap keeps the most recent logs of a pathological session.
const SESSION_PAGE_SIZE = 100;
const MAX_SESSION_PAGES = 50;
const SESSION_PAGES_PER_BATCH = 5;
// A session reopened within a minute reuses its loaded pages instead of
// downloading every page again. Calls that a running session makes in that
// minute show up on the first open after it.
const SESSION_STALE_TIME_MS = 60_000;

interface SessionPageRange {
  readonly first: number;
  readonly last: number;
}

interface SessionLogsBatch {
  readonly logs: LogEntry[];
  readonly total: number | undefined;
  readonly pageCount: number;
  readonly range: SessionPageRange;
}

const FIRST_SESSION_PAGE: SessionPageRange = { first: 1, last: 1 };

const emptySessionBatch = (range: SessionPageRange): SessionLogsBatch => ({ logs: [], total: 0, pageCount: 0, range });

async function fetchSessionLogsBatch(
  accessToken: string,
  sessionId: string,
  range: SessionPageRange,
  signal: AbortSignal,
): Promise<SessionLogsBatch> {
  const pages = await Promise.all(
    Array.from({ length: range.last - range.first + 1 }, (_, offset) =>
      sessionSpendLogsCall(accessToken, sessionId, {
        page: range.first + offset,
        page_size: SESSION_PAGE_SIZE,
        signal,
      }),
    ),
  );
  return {
    logs: pages.flatMap((page) => (Array.isArray(page) ? page : page.data ?? [])),
    total: pages[0]?.total,
    pageCount: Math.min(pages[0]?.total_pages ?? 1, MAX_SESSION_PAGES),
    range,
  };
}

function nextSessionPageRange(lastBatch: SessionLogsBatch, batches: SessionLogsBatch[]): SessionPageRange | undefined {
  const pageCount = batches[0].pageCount;
  const first = lastBatch.range.last + 1;
  if (first > pageCount) return undefined;
  return { first, last: Math.min(first + SESSION_PAGES_PER_BATCH - 1, pageCount) };
}

interface UseSessionLogsOptions {
  readonly sessionId: string | null | undefined;
  readonly accessToken: string | null | undefined;
  readonly enabled: boolean;
}

interface SessionLogs {
  readonly logs: LogEntry[];
  readonly total: number;
  readonly expected: number;
  readonly hasFirstPage: boolean;
  readonly isLoading: boolean;
}

export function useSessionLogs({ sessionId, accessToken, enabled }: UseSessionLogsOptions): SessionLogs {
  const isEnabled = Boolean(enabled && sessionId && accessToken);
  const queryOptions = {
    queryKey: ["sessionLogs", sessionId],
    queryFn: ({ pageParam, signal }: { pageParam: SessionPageRange; signal: AbortSignal }) =>
      sessionId && accessToken
        ? fetchSessionLogsBatch(accessToken, sessionId, pageParam, signal)
        : Promise.resolve(emptySessionBatch(pageParam)),
    initialPageParam: FIRST_SESSION_PAGE,
    getNextPageParam: nextSessionPageRange,
    enabled: isEnabled,
    staleTime: SESSION_STALE_TIME_MS,
    refetchOnWindowFocus: false,
  };
  const { data, hasNextPage, isFetching, isFetchNextPageError, fetchNextPage } = useInfiniteQuery(queryOptions);

  const isLoadingMore = Boolean(hasNextPage) && !isFetchNextPageError;
  const loadedBatches = data?.pages.length ?? 0;

  useEffect(() => {
    if (isEnabled && isLoadingMore && !isFetching) void fetchNextPage();
  }, [isEnabled, isLoadingMore, isFetching, loadedBatches, fetchNextPage]);

  const logs = useMemo<LogEntry[]>(() => {
    const rows = (data?.pages ?? []).flatMap((batch) => batch.logs);
    const uniqueRows = [...new Map(rows.map((row) => [row.request_id, row])).values()];
    return uniqueRows.map((row) => ({
      ...row,
      request_duration_ms: row.request_duration_ms ?? Date.parse(row.endTime) - Date.parse(row.startTime),
    }));
  }, [data]);

  const firstBatch = data?.pages[0];
  const total = firstBatch?.total ?? logs.length;
  return {
    logs,
    total,
    expected: Math.min(total, (firstBatch?.pageCount ?? 0) * SESSION_PAGE_SIZE),
    hasFirstPage: data !== undefined,
    isLoading: data === undefined ? isFetching : isLoadingMore,
  };
}
