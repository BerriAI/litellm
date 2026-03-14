import { useCallback, useEffect, useRef, useState } from "react";
import { DailyData } from "../types";

export interface PaginationProgress {
  currentPage: number;
  totalPages: number;
}

/** Delay between sequential page fetches (ms) to avoid overloading the backend. */
const PAGE_FETCH_DELAY_MS = 500;

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
 * Hook that auto-paginates daily activity endpoints, updating state after each
 * page so charts render progressively. Cancels on unmount, param changes, or
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

  const cancel = useCallback(() => {
    cancelledRef.current = true;
    setCancelled(true);
    setIsFetchingMore(false);
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

    const run = async () => {
      setLoading(true);
      setIsFetchingMore(false);
      setProgress({ currentPage: 1, totalPages: 1 });

      try {
        // Inject page=1 as the 4th argument.
        const argsWithPage = [...args.slice(0, 3), 1, ...args.slice(3)];
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
          await new Promise((resolve) =>
            setTimeout(resolve, PAGE_FETCH_DELAY_MS),
          );

          if (isStale()) return;

          const argsForPage = [...args.slice(0, 3), page, ...args.slice(3)];
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

          setData({
            results: accumulatedResults,
            metadata: accumulatedMetadata,
          });
          setProgress({ currentPage: page, totalPages });
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
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, fetchFn, ...args]);

  return { data, loading, isFetchingMore, progress, cancelled, cancel };
}
