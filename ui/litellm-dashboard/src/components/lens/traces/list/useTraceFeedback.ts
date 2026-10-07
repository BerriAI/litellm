import { useQueries } from "@tanstack/react-query";
import { chunk } from "es-toolkit";

import { useTracesApi } from "../api";
import type { TraceFeedbackSummary, TraceSummary } from "../types";

export const LOW_SCORE = 4;

export type TraceFeedbackState =
  | { status: "ready"; summary: Pick<TraceFeedbackSummary, "count" | "average" | "lowest"> }
  | { status: "pending" }
  | { status: "error" };

export const traceFeedbackKey = (accessToken: string) => ["traceFeedback", accessToken] as const;

/** A run some end user scored at or below the low-score threshold. */
export const isLowFeedback = (state: TraceFeedbackState | undefined): boolean =>
  state?.status === "ready" && state.summary.count > 0 && (state.summary.lowest ?? Infinity) <= LOW_SCORE;

export function useTraceFeedback(accessToken: string, runs: TraceSummary[], isActive: boolean) {
  const api = useTracesApi(accessToken);
  const batches = chunk(
    runs.map(({ trace_id, trace_ref }) => ({ trace_id, trace_ref: trace_ref ?? "" })),
    500,
  );
  const queries = useQueries({
    queries: batches.map((traces) => ({
      queryKey: [...traceFeedbackKey(accessToken), traces],
      queryFn: async () => {
        const summaries: unknown = await api.feedbackSummary(traces);
        if (!Array.isArray(summaries)) throw new Error("Unexpected feedback summary response");
        return summaries as TraceFeedbackSummary[];
      },
      enabled: isActive,
      staleTime: 15000,
      refetchInterval: isActive && api.live ? 15000 : false,
      retry: false,
    })),
  });
  return new Map<string, TraceFeedbackState>(
    batches.flatMap((traces, index) => {
      const query = queries[index];
      const summaries = new Map(query.data?.map((summary) => [summary.trace_ref || summary.trace_id, summary]));
      return traces.map((trace): [string, TraceFeedbackState] => {
        const key = trace.trace_ref || trace.trace_id;
        if (query.isError) return [key, { status: "error" }];
        if (query.isPending) return [key, { status: "pending" }];
        const summary = summaries.get(key);
        if (!summary) return [key, { status: "error" }];
        return [key, { status: "ready", summary }];
      });
    }),
  );
}
