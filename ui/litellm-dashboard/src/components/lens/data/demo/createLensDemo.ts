import { ApiError } from "@/lib/http/client";
import type { TracesApi } from "@/components/lens/traces/api";
import type { TimeWindow } from "@/components/shared/timeRange/timeRange";
import { filterRuns, RUN_INDEX } from "@/components/lens/traces/list/runSearch/runQuery";
import type { TraceHistogram, TraceSummary } from "@/components/lens/traces/types";
import { traceAgentNames } from "@/components/lens/traces/utils";
import type { LensServices } from "../LensServices";
import type { LensApi } from "../service";
import { createLensDemoData, type LensDemoData } from "./fixtures";

const notInDemo = (): Promise<never> =>
  Promise.reject(new ApiError("This item is not in the demo", 404, { detail: "This item is not in the demo" }));
const readOnly = (): Promise<never> =>
  Promise.reject(new ApiError("Demo data is read-only", 403, { detail: "Demo data is read-only" }));
const found = <T>(value: T | undefined): Promise<T> => (value === undefined ? notInDemo() : Promise.resolve(value));

function demoLensApi(data: LensDemoData): LensApi {
  const jobs = (lensId: string) => data.lenses.find((lens) => lens.id === lensId)?.jobs;
  return {
    scope: "demo",
    lenses: async () => ({ lenses: data.lenses, workers: [], tracing_enabled: true }),
    activity: async () => ({ traces: true, requests: false }),
    runs: (lensId, offset) => found(jobs(lensId)?.slice(offset)),
    run: (lensId, jobId) => found(jobs(lensId)?.find((job) => job.id === jobId)),
    execution: notInDemo,
    sample: notInDemo,
    agents: notInDemo,
    models: async () => ({ data: [] }),
    modelDetails: async () => ({ data: [] }),
    keys: notInDemo,
    keyInfo: notInDemo,
    saveLens: readOnly,
    startRun: readOnly,
    watchAll: async () => ({ watching: [], skipped: [] }),
    cancelRun: readOnly,
    reviewFinding: readOnly,
    registerWorker: readOnly,
    setWorkerBillingKey: readOnly,
    revokeWorker: readOnly,
    generateAnalysisKey: readOnly,
    deleteKeys: readOnly,
  };
}

const startedWithin = (run: TraceSummary, range: TimeWindow): boolean => {
  const startMs = Date.parse(run.start_time);
  return startMs >= range.startMs && startMs < range.endMs;
};

/** Runs per equal-width slice of the window, the shape the server's histogram returns. */
export function demoHistogram(runs: readonly TraceSummary[], range: TimeWindow, buckets: number): TraceHistogram {
  const width = (range.endMs - range.startMs) / buckets;
  const placed = runs.map((run) => ({
    index: Math.floor((Date.parse(run.start_time) - range.startMs) / width),
    failed: run.error_count > 0,
    agent: traceAgentNames(run)[0] ?? run.service,
  }));
  return {
    buckets: Array.from({ length: buckets }, (_, index) => {
      const hits = placed.filter((run) => run.index === index);
      const agents = [...new Set(hits.filter((run) => !run.failed).map((run) => run.agent))].sort();
      return {
        start_ms: range.startMs + index * width,
        end_ms: range.startMs + (index + 1) * width,
        total: hits.length,
        failed: hits.filter((run) => run.failed).length,
        agents: agents.map((agent) => ({
          agent,
          runs: hits.filter((run) => !run.failed && run.agent === agent).length,
        })),
      };
    }),
  };
}

function demoTracesApi(data: LensDemoData): TracesApi {
  const run = (traceId: string) => data.runs.find(({ trace }) => trace.summary.trace_id === traceId);
  const summaries = data.runs.map((item) => item.trace.summary);
  const matching = (range: TimeWindow, q: string) =>
    filterRuns(
      summaries.filter((summary) => startedWithin(summary, range)),
      q,
    );
  return {
    scope: "demo",
    live: false,
    handoff: (traceId, spanId) => {
      const found = run(traceId);
      const step = spanId ? found?.details.find((span) => span.span_id === spanId) : found;
      return { text: JSON.stringify(step, null, 2), copied: spanId ? "Step copied" : "Trace copied" };
    },
    list: async ({ startMs, endMs, q }) => ({ data: matching({ startMs, endMs }, q), next_cursor: null }),
    histogram: async (range, q, buckets) => demoHistogram(matching(range, q), range, buckets),
    values: async (field, contains, range) => {
      const read = field in RUN_INDEX.read ? RUN_INDEX.read[field as keyof typeof RUN_INDEX.read] : null;
      if (!read) return [];
      const needle = contains.toLowerCase();
      const values = matching(range, "").flatMap(read);
      return [...new Set(values)].filter((value) => value && value.toLowerCase().includes(needle)).sort();
    },
    anyRecorded: async () => data.runs.length > 0,
    trace: (traceId) => found(run(traceId)?.trace),
    span: (traceId, spanId) => found(run(traceId)?.details.find((span) => span.span_id === spanId)),
    spanError: async (traceId, spanId) => {
      const span = run(traceId)?.trace.spans.find((item) => item.span_id === spanId);
      return found(
        span && {
          span_id: span.span_id,
          message: span.error ?? "",
          total_chars: span.error?.length ?? 0,
          next_cursor: null,
        },
      );
    },
  };
}

export function createLensDemo(now = Date.now()): LensServices {
  const data = createLensDemoData(now);
  return { accessToken: "lens-demo", lens: demoLensApi(data), traces: demoTracesApi(data) };
}
