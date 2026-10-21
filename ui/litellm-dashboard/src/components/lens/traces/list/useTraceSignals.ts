import { useQueries, useQuery, useQueryClient, type Query, type QueryClient } from "@tanstack/react-query";
import { chunk } from "es-toolkit";

import { useTracesApi } from "../api";
import type { SignalFlag, TraceSignals, TraceSummary } from "../types";

export type TraceSignalState = { status: "ready"; signals: TraceSignals } | { status: "pending" } | { status: "error" };

const IDLE_POLL_MS = 15000;
const ACTIVE_POLL_MS = 2000;

const identity = ({ trace_id, trace_ref }: { trace_id: string; trace_ref?: string | null }) => ({
  trace_id,
  trace_ref: trace_ref ?? "",
});

const awaitingResult = (signals: TraceSignals): boolean =>
  signals.status === "unclassified" || signals.status === "pending";

export const signalPollInterval = (results: readonly TraceSignals[] | undefined): number =>
  results?.some(awaitingResult) ? ACTIVE_POLL_MS : IDLE_POLL_MS;

const pollWhile = (enabled: boolean, live: boolean) => (query: Query<TraceSignals[]>) =>
  enabled && live ? signalPollInterval(query.state.data) : false;

const signalKey = (result: { trace_id: string; trace_ref?: string | null }): string =>
  result.trace_ref || result.trace_id;

const cachedSignals = (client: QueryClient, accessToken: string): Map<string, TraceSignals> =>
  new Map(
    client
      .getQueriesData<TraceSignals[]>({ queryKey: ["traceSignals", accessToken] })
      .flatMap(([, data]) => data ?? [])
      .map((result) => [signalKey(result), result]),
  );

export const flaggedSignals = (signals?: TraceSignals): SignalFlag[] =>
  signals?.status === "classified" ? signals.flags ?? [] : [];

export const isFlagged = (state?: TraceSignalState): boolean =>
  state?.status === "ready" && flaggedSignals(state.signals).length > 0;

export function useTraceSignals(accessToken: string, runs: TraceSummary[], enabled: boolean) {
  const api = useTracesApi(accessToken);
  const client = useQueryClient();
  const batches = chunk(runs.map(identity), 500);
  const queries = useQueries({
    queries: batches.map((traces) => ({
      queryKey: ["traceSignals", accessToken, traces],
      queryFn: () => api.signals(traces),
      enabled,
      staleTime: ACTIVE_POLL_MS,
      refetchInterval: pollWhile(enabled, api.live),
      retry: false,
    })),
  });
  const known = cachedSignals(client, accessToken);
  return new Map<string, TraceSignalState>(
    batches.flatMap((traces, index) => {
      const query = queries[index];
      const results = new Map(query.data?.map((result) => [signalKey(result), result]));
      return traces.map((trace): [string, TraceSignalState] => {
        const key = signalKey(trace);
        const found = results.get(key) ?? known.get(key);
        if (found) return [key, { status: "ready", signals: found }];
        if (query.isError) return [key, { status: "error" }];
        if (query.isPending) return [key, { status: "pending" }];
        return [key, { status: "error" }];
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
    staleTime: ACTIVE_POLL_MS,
    refetchInterval: pollWhile(enabled, api.live),
    retry: false,
  };
  const query = useQuery<TraceSignals[]>(queryOptions);
  return enabled ? flaggedSignals(query.data?.[0]) : [];
}
