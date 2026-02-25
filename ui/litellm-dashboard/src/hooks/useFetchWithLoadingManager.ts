import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Manages fetch timing and loading display for API calls that affect the UI.
 *
 * Logic:
 * - On API call: Show loader immediately, fetch
 * - If more calls within 5 seconds of last fetch: Debounce. Wait 5 seconds from last fetch
 *   completion, then do one fetch with loader
 * - After 5 seconds of no new calls: Reset to idle (next call shows loader immediately)
 * - Never show loader if no calls are being made
 */
export function useFetchWithLoadingManager<T>(
  fetchFn: () => Promise<T>,
  options?: { debounceMs?: number }
) {
  const { debounceMs = 5000 } = options ?? {};
  const [loading, setLoading] = useState(false);
  const lastFetchEndRef = useRef(0);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fetchInProgressRef = useRef(false);

  const requestFetch = useCallback((): Promise<T> => {
    const now = Date.now();
    const timeSinceLastFetch = now - lastFetchEndRef.current;
    const isFirstCall = lastFetchEndRef.current === 0 && !fetchInProgressRef.current;
    const fetchInProgress = fetchInProgressRef.current;

    const executeFetch = async (): Promise<T> => {
      fetchInProgressRef.current = true;
      setLoading(true);
      try {
        const result = await fetchFn();
        return result;
      } finally {
        fetchInProgressRef.current = false;
        lastFetchEndRef.current = Date.now();
        if (!debounceTimerRef.current) {
          setLoading(false);
        }
      }
    };

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
      debounceTimerRef.current = null;
    }

    if (fetchInProgress) {
      setLoading(true);
      const delay = debounceMs;
      return new Promise<T>((resolve, reject) => {
        debounceTimerRef.current = setTimeout(async () => {
          debounceTimerRef.current = null;
          try {
            const result = await executeFetch();
            resolve(result);
          } catch (e) {
            reject(e);
          }
        }, delay);
      });
    }
    if (isFirstCall || timeSinceLastFetch >= debounceMs) {
      return executeFetch();
    } else {
      setLoading(true);
      const delay = debounceMs - timeSinceLastFetch;
      return new Promise<T>((resolve, reject) => {
        debounceTimerRef.current = setTimeout(async () => {
          debounceTimerRef.current = null;
          try {
            const result = await executeFetch();
            resolve(result);
          } catch (e) {
            reject(e);
          }
        }, delay);
      });
    }
  }, [fetchFn, debounceMs]);

  useEffect(() => {
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, []);

  return { loading, requestFetch };
}
