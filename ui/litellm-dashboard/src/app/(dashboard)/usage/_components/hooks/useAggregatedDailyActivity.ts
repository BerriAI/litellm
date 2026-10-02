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
}

interface SettledFetch {
  key: string;
  data: DailyActivityAggregatedResponse;
  failed: boolean;
}

export function useAggregatedDailyActivity({ fetch, enabled, deps }: Options): Result {
  const [settled, setSettled] = useState<SettledFetch | null>(null);
  const requestIdRef = useRef(0);
  const fetchRef = useRef(fetch);

  useEffect(() => {
    fetchRef.current = fetch;
  });

  const depsKey = JSON.stringify(deps);

  useEffect(() => {
    if (!enabled) return;

    const requestId = ++requestIdRef.current;
    const isStale = () => requestIdRef.current !== requestId;

    fetchRef
      .current()
      .then((response) => {
        if (isStale()) return;
        setSettled({ key: depsKey, data: response, failed: false });
      })
      .catch((error) => {
        if (isStale()) return;
        console.error("Error fetching daily activity:", error);
        setSettled({ key: depsKey, data: EMPTY_DAILY_ACTIVITY_RESPONSE, failed: true });
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
  };
}
