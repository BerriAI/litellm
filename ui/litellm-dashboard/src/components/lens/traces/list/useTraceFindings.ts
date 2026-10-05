import { useQueries } from "@tanstack/react-query";
import { chunk } from "es-toolkit";

import { useTracesApi } from "../api";
import type { TraceSummary } from "../types";

export type TraceFindingState = { status: "ready"; count: number | null } | { status: "pending" } | { status: "error" };

export function useTraceFindings(accessToken: string, runs: TraceSummary[], isActive: boolean, canViewFindings = true) {
  const api = useTracesApi(accessToken);
  const enabled = isActive && canViewFindings;
  const batches = chunk(
    runs.map(({ trace_id, trace_ref }) => ({ trace_id, trace_ref: trace_ref ?? "" })),
    500,
  );
  const queries = useQueries({
    queries: batches.map((traces) => ({
      queryKey: ["traceFindings", accessToken, traces],
      queryFn: () => api.findings(traces),
      enabled,
      staleTime: 15000,
      refetchInterval: enabled && api.live ? 15000 : false,
      retry: false,
    })),
  });
  return new Map<string, TraceFindingState>(
    batches.flatMap((traces, index) => {
      const query = queries[index];
      const counts = new Map(query.data?.map((result) => [result.trace_ref || result.trace_id, result.finding_count]));
      return traces.map((trace): [string, TraceFindingState] => {
        const key = trace.trace_ref || trace.trace_id;
        if (query.isError) return [key, { status: "error" }];
        if (query.isPending) return [key, { status: "pending" }];
        if (!counts.has(key)) return [key, { status: "error" }];
        return [key, { status: "ready", count: counts.get(key) ?? null }];
      });
    }),
  );
}
