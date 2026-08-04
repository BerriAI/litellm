import { useQuery } from "@tanstack/react-query";

import { autoRouterBenchmarksCall } from "@/components/networking";

import type { AutoRouterBenchmarksResponse } from "./autoRouterBenchmarks";

export const BENCHMARKS_WINDOW_DAYS = 30;

const isoDate = (date: Date): string => date.toISOString().slice(0, 10);

/**
 * The window is computed once per mount rather than per render, so the query key
 * stays stable and the request is not refired every time the parent re-renders.
 */
export const benchmarksWindow = (now: Date): { start: string; end: string } => ({
  start: isoDate(new Date(now.getTime() - BENCHMARKS_WINDOW_DAYS * 24 * 60 * 60 * 1000)),
  end: isoDate(now),
});

export const useAutoRouterBenchmarks = (accessToken: string | null, window: { start: string; end: string }) =>
  useQuery<AutoRouterBenchmarksResponse>({
    queryKey: ["autoRouterBenchmarks", window.start, window.end],
    queryFn: async () => await autoRouterBenchmarksCall(accessToken!, window.start, window.end),
    enabled: Boolean(accessToken),
    retry: false,
  });
