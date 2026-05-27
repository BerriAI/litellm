import { useCallback, useEffect, useRef, useState } from "react";
import { DailyData } from "../types";

export interface PaginationProgress {
  currentPage: number;
  totalPages: number;
}

/** Delay between sequential page fetches (ms) to avoid overloading the backend. */
const PAGE_FETCH_DELAY_MS = 300;

/** Number of pages to accumulate before flushing to React state (reduces re-renders). */
const RENDER_BATCH_SIZE = 3;

/** The metadata fields returned by the daily activity API that should be summed across pages. */
const SUMMABLE_METADATA_KEYS = [
  "total_spend",
  "total_prompt_tokens",
  "total_completion_tokens",
  "total_tokens",
  "total_api_requests",
  "total_successful_requests",
  "total_failed_requests",
  "total_cache_read_input_tokens",
  "total_cache_creation_input_tokens",
] as const;

/** The per-row metric fields that should be summed when two rows share a date. */
const SUMMABLE_ROW_METRIC_KEYS = [
  "spend",
  "prompt_tokens",
  "completion_tokens",
  "total_tokens",
  "api_requests",
  "successful_requests",
  "failed_requests",
  "cache_read_input_tokens",
  "cache_creation_input_tokens",
] as const;

interface DailyActivityResponse {
  results: DailyData[];
  metadata: Record<string, any>;
}

type FetchPageFn = (...args: any[]) => Promise<DailyActivityResponse>;

interface UsePaginatedDailyActivityParams {
  /** The API call function (e.g., userDailyActivityCall). */
  fetchFn: FetchPageFn;
  /** Arguments to pass to fetchFn: [accessToken, startTime, endTime, ...extraArgs]. Page is injected by the hook at index 3. */
  args: any[];
  /** Whether the hook should fetch. Set to false to disable. */
  enabled: boolean;
}

interface UsePaginatedDailyActivityReturn {
  data: DailyActivityResponse;
  loading: boolean;
  isFetchingMore: boolean;
  progress: PaginationProgress;
  cancelled: boolean;
  cancel: () => void;
}

const EMPTY_DATA: DailyActivityResponse = {
  results: [],
  metadata: {
    total_spend: 0,
    total_prompt_tokens: 0,
    total_completion_tokens: 0,
    total_tokens: 0,
    total_api_requests: 0,
    total_successful_requests: 0,
    total_failed_requests: 0,
    total_cache_read_input_tokens: 0,
    total_cache_creation_input_tokens: 0,
    total_pages: 1,
    has_more: false,
    page: 1,
  },
};

function sumMetadata(
  a: Record<string, any>,
  b: Record<string, any>,
): Record<string, any> {
  const result = { ...a };
  for (const key of SUMMABLE_METADATA_KEYS) {
    result[key] = (a[key] || 0) + (b[key] || 0);
  }
  return result;
}

function sumRowMetrics(
  a: Record<string, any>,
  b: Record<string, any>,
): Record<string, any> {
  const result = { ...a };
  for (const key of SUMMABLE_ROW_METRIC_KEYS) {
    result[key] = (a[key] || 0) + (b[key] || 0);
  }
  return result;
}

/**
 * Merge two breakdown sub-maps (e.g. models, api_keys) keyed by entity id.
 * When the same entity appears in both, metrics are summed and the nested
 * api_key_breakdown is recursively merged. metadata from the first row wins.
 */
function mergeBreakdownSubMap(
  a: Record<string, any> | undefined,
  b: Record<string, any> | undefined,
): Record<string, any> {
  if (!a) return { ...(b || {}) };
  if (!b) return { ...a };
  const result: Record<string, any> = { ...a };
  for (const [key, bEntry] of Object.entries(b)) {
    const aEntry = result[key];
    if (!aEntry) {
      result[key] = bEntry;
      continue;
    }
    result[key] = {
      ...aEntry,
      metrics: sumRowMetrics(aEntry.metrics || {}, bEntry.metrics || {}),
      metadata: aEntry.metadata ?? bEntry.metadata,
      api_key_breakdown: mergeBreakdownSubMap(
        aEntry.api_key_breakdown,
        bEntry.api_key_breakdown,
      ),
    };
  }
  return result;
}

const BREAKDOWN_SUBMAPS = [
  "models",
  "model_groups",
  "mcp_servers",
  "providers",
  "api_keys",
  "entities",
  "endpoints",
] as const;

/**
 * Merge the full breakdown object of two DailyData rows that share a date.
 * Each known sub-map is merged independently; unknown sub-maps from either
 * side are passed through so we never drop fields the backend adds later.
 */
export function mergeBreakdowns(
  a: Record<string, any> | undefined,
  b: Record<string, any> | undefined,
): Record<string, any> {
  const merged: Record<string, any> = { ...(a || {}), ...(b || {}) };
  for (const key of BREAKDOWN_SUBMAPS) {
    merged[key] = mergeBreakdownSubMap(a?.[key], b?.[key]);
  }
  return merged;
}

/**
 * Collapse paginated DailyData results so rows with the same date are
 * merged into one entry (metrics summed, breakdowns merged). Without this,
 * a paginated backend response for a single-day window emits one chart bar
 * per page instead of one bar per actual day. Order is preserved by
 * first-seen date.
 */
export function mergeResultsByDate(rows: DailyData[]): DailyData[] {
  const byDate = new Map<string, DailyData>();
  for (const row of rows) {
    const existing = byDate.get(row.date);
    if (!existing) {
      byDate.set(row.date, row);
      continue;
    }
    byDate.set(row.date, {
      date: row.date,
      metrics: sumRowMetrics(existing.metrics, row.metrics) as DailyData["metrics"],
      breakdown: mergeBreakdowns(existing.breakdown, row.breakdown) as DailyData["breakdown"],
    });
  }
  return Array.from(byDate.values());
}

/**
 * Hook that auto-paginates daily activity endpoints, updating state in batches
 * so charts render progressively. Cancels on unmount, param changes, or
 * manual cancel().
 *
 * The `args` array should contain every argument the fetchFn expects EXCEPT
 * the `page` parameter. The hook injects `page` as the 4th argument (index 3),
 * matching the signature of all daily activity calls:
 *   (accessToken, startTime, endTime, page, ...rest)
 */
export function usePaginatedDailyActivity({
  fetchFn,
  args,
  enabled,
}: UsePaginatedDailyActivityParams): UsePaginatedDailyActivityReturn {
  const [data, setData] = useState<DailyActivityResponse>(EMPTY_DATA);
  const [loading, setLoading] = useState(false);
  const [isFetchingMore, setIsFetchingMore] = useState(false);
  const [progress, setProgress] = useState<PaginationProgress>({
    currentPage: 0,
    totalPages: 0,
  });
  const [cancelled, setCancelled] = useState(false);

  const fetchIdRef = useRef(0);
  const cancelledRef = useRef(false);
  const delayTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep args in a ref so the effect can always read the latest values
  // without needing them in the dependency array.
  const argsRef = useRef(args);
  argsRef.current = args;

  // Stable serialised key so the effect only re-runs when the arg *values* change.
  const argsKey = JSON.stringify(args);

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    setCancelled(true);
    setIsFetchingMore(false);
    if (delayTimerRef.current !== null) {
      clearTimeout(delayTimerRef.current);
      delayTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!enabled) {
      setData(EMPTY_DATA);
      setLoading(false);
      setIsFetchingMore(false);
      setProgress({ currentPage: 0, totalPages: 0 });
      setCancelled(false);
      return;
    }

    const currentFetchId = ++fetchIdRef.current;
    cancelledRef.current = false;
    setCancelled(false);

    const isStale = () =>
      fetchIdRef.current !== currentFetchId || cancelledRef.current;

    /** Cancellable delay that clears itself on cleanup. */
    const delay = (ms: number) =>
      new Promise<void>((resolve) => {
        delayTimerRef.current = setTimeout(() => {
          delayTimerRef.current = null;
          resolve();
        }, ms);
      });

    const run = async () => {
      const currentArgs = argsRef.current;
      setLoading(true);
      setIsFetchingMore(false);
      setProgress({ currentPage: 1, totalPages: 1 });

      try {
        // Inject page=1 as the 4th argument.
        const argsWithPage = [...currentArgs.slice(0, 3), 1, ...currentArgs.slice(3)];
        const firstPage = await fetchFn(...argsWithPage);

        if (isStale()) return;

        // Dedupe rows that share a date within the first page too -- e.g. the
        // backend _adjust_dates_for_timezone may emit a UTC ghost-bar for the
        // same local-day window.
        const firstPageDeduped: DailyActivityResponse = {
          ...firstPage,
          results: mergeResultsByDate(firstPage.results),
        };
        setData(firstPageDeduped);

        const totalPages = firstPage.metadata?.total_pages || 1;

        setProgress({ currentPage: 1, totalPages });

        if (totalPages <= 1) {
          setLoading(false);
          return;
        }

        // More pages -- start fetching sequentially.
        setLoading(false);
        setIsFetchingMore(true);

        let accumulatedResults = [...firstPageDeduped.results];
        let accumulatedMetadata = { ...firstPage.metadata };

        for (let page = 2; page <= totalPages; page++) {
          if (isStale()) return;

          // Small delay to avoid overwhelming the backend.
          await delay(PAGE_FETCH_DELAY_MS);

          if (isStale()) return;

          const argsForPage = [...currentArgs.slice(0, 3), page, ...currentArgs.slice(3)];
          const pageData = await fetchFn(...argsForPage);

          if (isStale()) return;

          // Merge rows that share the same date (one bar per actual day,
          // not one bar per paginated page).
          accumulatedResults = mergeResultsByDate([
            ...accumulatedResults,
            ...pageData.results,
          ]);
          accumulatedMetadata = sumMetadata(
            accumulatedMetadata,
            pageData.metadata,
          );
          accumulatedMetadata.total_pages = totalPages;
          accumulatedMetadata.has_more = page < totalPages;
          accumulatedMetadata.page = page;

          // Flush accumulated data and progress to React state every
          // RENDER_BATCH_SIZE pages (or on the final page) to avoid
          // expensive per-page re-renders. Progress and data are updated
          // together so the counter never appears to decrement.
          const isLastPage = page === totalPages;
          const isBatchBoundary = (page - 1) % RENDER_BATCH_SIZE === 0;
          if (isLastPage || isBatchBoundary) {
            setData({
              results: accumulatedResults,
              metadata: accumulatedMetadata,
            });
            setProgress({ currentPage: page, totalPages });
          }
        }

        setIsFetchingMore(false);
      } catch (error) {
        if (!isStale()) {
          console.error("Error fetching daily activity:", error);
          setLoading(false);
          setIsFetchingMore(false);
        }
      }
    };

    run();

    return () => {
      fetchIdRef.current++;
      if (delayTimerRef.current !== null) {
        clearTimeout(delayTimerRef.current);
        delayTimerRef.current = null;
      }
    };
    // argsKey is a stable JSON string so the effect only re-fires when arg values change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, fetchFn, argsKey]);

  return { data, loading, isFetchingMore, progress, cancelled, cancel };
}
