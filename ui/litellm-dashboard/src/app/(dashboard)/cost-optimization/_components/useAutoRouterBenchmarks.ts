import { useEffect, useMemo, useState } from "react";

import { autoRouterBenchmarksCall, AutoRouterBenchmarksResponse } from "@/components/networking";

const THIRTY_DAYS_MS = 30 * 24 * 60 * 60 * 1000;

const toIsoDate = (d: Date): string => d.toISOString().slice(0, 10);

export interface AutoRouterBenchmarksState {
  data: AutoRouterBenchmarksResponse | null;
  loading: boolean;
  error: string | null;
}

interface Resolved {
  key: string;
  data: AutoRouterBenchmarksResponse | null;
  error: string | null;
}

export const useAutoRouterBenchmarks = (accessToken: string | null): AutoRouterBenchmarksState => {
  const startDate = useMemo(() => toIsoDate(new Date(new Date().getTime() - THIRTY_DAYS_MS)), []);
  const endDate = useMemo(() => toIsoDate(new Date()), []);

  const requestKey = accessToken ? `${accessToken}:${startDate}:${endDate}` : "";
  const [resolved, setResolved] = useState<Resolved | null>(null);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    let cancelled = false;
    autoRouterBenchmarksCall(accessToken, startDate, endDate)
      .then((data) => {
        if (!cancelled) setResolved({ key: requestKey, data, error: null });
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setResolved({ key: requestKey, data: null, error: e instanceof Error ? e.message : "Failed to load" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [accessToken, startDate, endDate, requestKey]);

  if (!accessToken) {
    return { data: null, loading: false, error: null };
  }
  if (resolved?.key !== requestKey) {
    return { data: null, loading: true, error: null };
  }
  return { data: resolved.data, loading: false, error: resolved.error };
};
