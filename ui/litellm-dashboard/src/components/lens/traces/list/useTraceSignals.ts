import { useQueries, useQuery } from "@tanstack/react-query";
import { chunk } from "es-toolkit";

import { useTracesApi } from "../api";
import type { SignalFlag, TraceSignals, TraceSummary } from "../types";

export type TraceSignalState = { status: "ready"; signals: TraceSignals } | { status: "pending" } | { status: "error" };

const POLL_MS = 15000;

const identity = ({ trace_id, trace_ref }: { trace_id: string; trace_ref?: string | null }) => ({
  trace_id,
  trace_ref: trace_ref ?? "",
});

export const flaggedSignals = (signals?: TraceSignals): SignalFlag[] =>
  signals?.status === "classified" ? signals.flags ?? [] : [];

export const isFlagged = (state?: TraceSignalState): boolean =>
  state?.status === "ready" && flaggedSignals(state.signals).length > 0;

export function useTraceSignals(accessToken: string, runs: TraceSummary[], enabled: boolean) {
  const api = useTracesApi(accessToken);
  const batches = chunk(runs.map(identity), 500);
  const queries = useQueries({
    queries: batches.map((traces) => ({
      queryKey: ["traceSignals", accessToken, traces],
      queryFn: () => api.signals(traces),
      enabled,
      staleTime: POLL_MS,
      refetchInterval: enabled && api.live ? POLL_MS : false,
      retry: false,
    })),
  });
  return new Map<string, TraceSignalState>(
    batches.flatMap((traces, index) => {
      const query = queries[index];
      const results = new Map(query.data?.map((result) => [result.trace_ref || result.trace_id, result]));
      return traces.map((trace): [string, TraceSignalState] => {
        const key = trace.trace_ref || trace.trace_id;
        const found = results.get(key);
        if (query.isError) return [key, { status: "error" }];
        if (query.isPending) return [key, { status: "pending" }];
        if (!found) return [key, { status: "error" }];
        return [key, { status: "ready", signals: found }];
      });
    }),
  );
}

export function useTraceSignalFlags(
  accessToken: string,
  trace: { trace_id: string; trace_ref?: string | null },
  enabled: boolean,
): SignalFlag[] {
  const api = useTracesApi(accessToken);
  const traces = [identity(trace)];
  const queryOptions = {
    queryKey: ["traceSignals", accessToken, traces],
    queryFn: () => api.signals(traces),
    enabled,
    staleTime: POLL_MS,
    refetchInterval: enabled && api.live ? POLL_MS : (false as const),
    retry: false,
  };
  const query = useQuery<TraceSignals[]>(queryOptions);
  return enabled ? flaggedSignals(query.data?.[0]) : [];
}
