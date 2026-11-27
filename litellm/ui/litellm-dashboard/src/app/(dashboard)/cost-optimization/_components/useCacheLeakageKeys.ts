import { useEffect, useRef, useState } from "react";

import { cacheLeakageKeysCall } from "@/components/networking";
import type { KeySpendActivityRow } from "@/components/UsagePage/dailyActivityApi";
import type { DailyActivityRange } from "./useDailyActivityRange";

interface CacheLeakageKeysResult {
  rows: KeySpendActivityRow[];
  loading: boolean;
  failed: boolean;
}

interface SettledKeys {
  key: string;
  rows: KeySpendActivityRow[];
  failed: boolean;
}

export const useCacheLeakageKeys = (range: DailyActivityRange, enabled: boolean): CacheLeakageKeysResult => {
  const { accessToken, startTime, endTime, userId, apiKey } = range.scope;
  const [settled, setSettled] = useState<SettledKeys | null>(null);
  const requestIdRef = useRef(0);

  const hasTimeRange = !!startTime && !!endTime;
  const scopeReady = enabled && !!accessToken && hasTimeRange;
  const scopeKey = scopeReady ? JSON.stringify([accessToken, startTime, endTime, userId, apiKey]) : null;

  useEffect(() => {
    if (!scopeKey) return;
    if (!accessToken || !startTime || !endTime) return;

    const requestId = ++requestIdRef.current;
    const isStale = () => requestIdRef.current !== requestId;

    const request = {
      accessToken,
      startTime,
      endTime,
      entityIds: userId ? [userId] : null,
      apiKey,
      includeCurrentUtcDay: true,
    };
    cacheLeakageKeysCall(request)
      .then((response) => {
        if (isStale()) return;
        setSettled({ key: scopeKey, rows: response.api_keys, failed: false });
      })
      .catch((error) => {
        if (isStale()) return;
        console.error("Failed to fetch cache leakage keys:", error);
        setSettled({ key: scopeKey, rows: [], failed: true });
      });

    return () => {
      requestIdRef.current++;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- scopeKey serializes the scope
  }, [scopeKey]);

  const current = scopeKey !== null && settled?.key === scopeKey ? settled : null;
  return {
    rows: current?.rows ?? [],
    loading: scopeKey !== null && current === null,
    failed: current?.failed ?? false,
  };
};
