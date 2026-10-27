import { useEffect, useRef, useState } from "react";
import {
  EMPTY_DAILY_ACTIVITY_RESPONSE,
  type DailyActivityAggregatedResponse,
} from "@/components/UsagePage/dailyActivityApi";

interface Options {
  fetch: () => Promise<DailyActivityAggregatedResponse>;
  enabled: boolean;
  deps: readonly unknown[];
}

interface Result {
  data: DailyActivityAggregatedResponse;
  loading: boolean;
  failed: boolean;
  errorMessage: string | null;
  reload: () => void;
}

interface SettledFetch {
  key: string;
  data: DailyActivityAggregatedResponse;
  failed: boolean;
  errorMessage: string | null;
}

const settledFetch = (
  key: string,
  data: DailyActivityAggregatedResponse,
  errorMessage: string | null,
): SettledFetch => ({
  key,
  data,
  failed: errorMessage !== null,
  errorMessage,
});

export function useAggregatedDailyActivity({ fetch, enabled, deps }: Options): Result {
  const [settled, setSettled] = useState<SettledFetch | null>(null);
  const [attempt, setAttempt] = useState(0);
  const requestIdRef = useRef(0);
  const fetchRef = useRef(fetch);

  useEffect(() => {
    fetchRef.current = fetch;
  });

  const depsKey = JSON.stringify([attempt, ...deps]);

  useEffect(() => {
    if (!enabled) return;

    const requestId = ++requestIdRef.current;
    const isStale = () => requestIdRef.current !== requestId;

    fetchRef
      .current()
      .then((response) => {
        if (isStale()) return;
        setSettled(settledFetch(depsKey, response, null));
      })
      .catch((error) => {
        if (isStale()) return;
        console.error("Error fetching daily activity:", error);
        const message = error instanceof Error ? error.message : String(error);
        setSettled(settledFetch(depsKey, EMPTY_DAILY_ACTIVITY_RESPONSE, message));
      });

    return () => {
      requestIdRef.current++;
    };
  }, [enabled, depsKey]);

  const current = enabled && settled?.key === depsKey ? settled : null;
  return {
    data: current?.data ?? EMPTY_DAILY_ACTIVITY_RESPONSE,
    loading: enabled && current === null,
    failed: current?.failed ?? false,
    errorMessage: current?.errorMessage ?? null,
    reload: () => setAttempt((n) => n + 1),
  };
}
