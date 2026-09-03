import { useCallback, useEffect, useRef, useState } from "react";
import {
  BreakdownMetrics,
  DailyData,
  KeyMetricWithMetadata,
  MetricWithMetadata,
  SpendMetrics,
} from "@/components/UsagePage/types";

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
  "total_flat_cost",
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
  /**
   * Optional single-shot endpoint returning the whole range at once (e.g.
   * teamDailyActivityAggregatedCall). Called with `args` as-is (no page).
   * On success pagination is skipped entirely; on failure the hook falls
   * back to the paginated flow.
   */
  aggregatedFetchFn?: (...args: any[]) => Promise<DailyActivityResponse>;
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

/**
 * Combine two pages of metadata. Only keys in SUMMABLE_METADATA_KEYS are added; anything
 * else keeps the first page's value, so a total the backend adds later is silently frozen
 * at page 1 until it is listed above. Exported so that contract can be tested directly.
 */
export function sumMetadata(a: Record<string, any>, b: Record<string, any>): Record<string, any> {
  const result = { ...a };
  for (const key of SUMMABLE_METADATA_KEYS) {
    result[key] = (a[key] || 0) + (b[key] || 0);
  }
  return result;
}

/**
 * Sum the union of numeric metric keys so a metric column added to the backend
 * later is summed automatically instead of silently frozen at one page's value
 * (the drift hazard SUMMABLE_METADATA_KEYS documents above).
 */
const addMetrics = (a: SpendMetrics, b: SpendMetrics): SpendMetrics =>
  Object.fromEntries(
    Array.from(new Set([...Object.keys(a), ...Object.keys(b)])).map((key) => {
      const left = a[key as keyof SpendMetrics];
      const right = b[key as keyof SpendMetrics];
      if (typeof left !== "number" && typeof right !== "number") return [key, left ?? right];
      return [key, (typeof left === "number" ? left : 0) + (typeof right === "number" ? right : 0)];
    }),
  ) as unknown as SpendMetrics;

const mergeBucketMaps = <T>(
  a: Record<string, T> | undefined,
  b: Record<string, T> | undefined,
  mergeEntry: (left: T, right: T) => T,
): Record<string, T> => {
  const left = a ?? {};
  const right = b ?? {};
  return Object.fromEntries(
    Array.from(new Set([...Object.keys(left), ...Object.keys(right)])).map((key) => {
      const leftEntry = left[key];
      const rightEntry = right[key];
      if (leftEntry === undefined) return [key, rightEntry];
      if (rightEntry === undefined) return [key, leftEntry];
      return [key, mergeEntry(leftEntry, rightEntry)];
    }),
  );
};

const mergeKeyMetric = (a: KeyMetricWithMetadata, b: KeyMetricWithMetadata): KeyMetricWithMetadata => ({
  ...a,
  metrics: addMetrics(a.metrics, b.metrics),
});

const mergeMetricWithMetadata = (a: MetricWithMetadata, b: MetricWithMetadata): MetricWithMetadata => ({
  ...a,
  metrics: addMetrics(a.metrics, b.metrics),
  api_key_breakdown: mergeBucketMaps(a.api_key_breakdown, b.api_key_breakdown, mergeKeyMetric),
});

const mergeBreakdown = (a: BreakdownMetrics, b: BreakdownMetrics): BreakdownMetrics => ({
  models: mergeBucketMaps(a.models, b.models, mergeMetricWithMetadata),
  model_groups: mergeBucketMaps(a.model_groups, b.model_groups, mergeMetricWithMetadata),
  mcp_servers: mergeBucketMaps(a.mcp_servers, b.mcp_servers, mergeMetricWithMetadata),
  providers: mergeBucketMaps(a.providers, b.providers, mergeMetricWithMetadata),
  api_keys: mergeBucketMaps(a.api_keys, b.api_keys, mergeKeyMetric),
  entities: mergeBucketMaps(a.entities, b.entities, mergeMetricWithMetadata),
  ...(a.endpoints || b.endpoints
    ? { endpoints: mergeBucketMaps(a.endpoints, b.endpoints, mergeMetricWithMetadata) }
    : {}),
});

/**
 * The backend paginates over raw rows and re-groups per page, so a date whose
 * rows span pages arrives as one partial DailyData per page. Merge by date so
 * consumers never see the same date twice (LIT-5818: each day rendered as N
 * partial bars). Exported so the contract can be tested directly.
 */
export function mergeDailyResults(existing: readonly DailyData[], incoming: readonly DailyData[]): DailyData[] {
  return incoming.reduce<DailyData[]>(
    (acc, day) => {
      const index = acc.findIndex((existingDay) => existingDay.date === day.date);
      if (index === -1) return [...acc, day];
      return acc.map((existingDay, i) =>
        i === index
          ? {
              ...existingDay,
              metrics: addMetrics(existingDay.metrics, day.metrics),
              breakdown: mergeBreakdown(existingDay.breakdown, day.breakdown),
            }
          : existingDay,
      );
    },
    [...existing],
  );
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
  aggregatedFetchFn,
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

    const isStale = () => fetchIdRef.current !== currentFetchId || cancelledRef.current;

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

      if (aggregatedFetchFn) {
        try {
          const aggregated = await aggregatedFetchFn(...currentArgs);
          if (isStale()) return;
          setData(aggregated);
          setProgress({ currentPage: 1, totalPages: 1 });
          setLoading(false);
          return;
        } catch (error) {
          if (isStale()) return;
          console.error("Aggregated daily activity failed, falling back to pagination:", error);
        }
      }

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

        let accumulatedResults = mergeDailyResults([], firstPage.results);
        let accumulatedMetadata = { ...firstPage.metadata };

        for (let page = 2; page <= totalPages; page++) {
          if (isStale()) return;

          // Small delay to avoid overwhelming the backend.
          await delay(PAGE_FETCH_DELAY_MS);

          if (isStale()) return;

          const argsForPage = [...currentArgs.slice(0, 3), page, ...currentArgs.slice(3)];
          const pageData = await fetchFn(...argsForPage);

          if (isStale()) return;

          accumulatedResults = mergeDailyResults(accumulatedResults, pageData.results);
          accumulatedMetadata = sumMetadata(accumulatedMetadata, pageData.metadata);
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
  }, [enabled, fetchFn, aggregatedFetchFn, argsKey]);

  return { data, loading, isFetchingMore, progress, cancelled, cancel };
}
