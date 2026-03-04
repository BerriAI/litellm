import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { getPerformanceSummaryCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export interface LatencyStats {
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
}

export interface PerModelStats {
  model: string;
  overhead: LatencyStats | null;
  llm_api: LatencyStats | null;
  total: LatencyStats | null;
  sample_count: number;
}

export interface PerformanceIssue {
  severity: "critical" | "warning" | "info";
  title: string;
  description: string;
  fix: string;
  fix_snippet: string | null;
}

export interface PerformanceSummaryResponse {
  debug_flags: {
    is_detailed_debug: boolean;
    log_level: string;
    detailed_timing_enabled: boolean;
  };
  workers: {
    cpu_count: number;
    num_workers: number;
    cpu_percent: number | null;
  };
  connection_pools: {
    in_flight_requests: number | null;
    db: {
      connected: boolean;
      pool_limit: number;
      pool_timeout_seconds: number;
    };
    redis: {
      enabled: boolean;
      max_connections?: number | null;
    };
    http: {
      aiohttp_limit: number;
      aiohttp_active: number | null;
      aiohttp_pct: number | null;
    };
  };
  latency: {
    overhead: LatencyStats | null;
    llm_api: LatencyStats | null;
    pre_processing: LatencyStats | null;
    post_processing: LatencyStats | null;
    total: LatencyStats | null;
    overhead_pct_of_total: number | null;
    sample_count: number;
    overhead_histogram: { bucket: string; count: number }[];
  };
  per_model: PerModelStats[];
  issues: PerformanceIssue[];
}

export const usePerformanceSummary = (): UseQueryResult<PerformanceSummaryResponse> => {
  const { accessToken } = useAuthorized();
  return useQuery<PerformanceSummaryResponse>({
    queryKey: ["performanceSummary"],
    queryFn: async () => getPerformanceSummaryCall(accessToken!),
    enabled: Boolean(accessToken),
    refetchInterval: 10_000,
  });
};
