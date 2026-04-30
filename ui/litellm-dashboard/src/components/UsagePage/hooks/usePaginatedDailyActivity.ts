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

        setData(firstPage);

        const totalPages = firstPage.metadata?.total_pages || 1;

        setProgress({ currentPage: 1, totalPages });

        if (totalPages <= 1) {
          setLoading(false);
          return;
        }

        // More pages — start fetching sequentially.
        setLoading(false);
        setIsFetchingMore(true);

        let accumulatedResults = [...firstPage.results];
        let accumulatedMetadata = { ...firstPage.metadata };

        for (let page = 2; page <= totalPages; page++) {
          if (isStale()) return;

          // Small delay to avoid overwhelming the backend.
          await delay(PAGE_FETCH_DELAY_MS);

          if (isStale()) return;

          const argsForPage = [...currentArgs.slice(0, 3), page, ...currentArgs.slice(3)];
          const pageData = await fetchFn(...argsForPage);

          if (isStale()) return;

          accumulatedResults = [...accumulatedResults, ...pageData.results];
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
