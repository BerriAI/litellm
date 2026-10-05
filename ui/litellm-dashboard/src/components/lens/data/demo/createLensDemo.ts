import { ApiError } from "@/lib/http/client";
import type { TracesApi } from "@/components/lens/traces/api";
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

function demoTracesApi(data: LensDemoData): TracesApi {
  const run = (traceId: string) => data.runs.find(({ trace }) => trace.summary.trace_id === traceId);
  return {
    live: false,
    handoff: (traceId, spanId) => {
      const found = run(traceId);
      const step = spanId ? found?.details.find((span) => span.span_id === spanId) : found;
      return { text: JSON.stringify(step, null, 2), copied: spanId ? "Step copied" : "Trace copied" };
    },
    list: async ({ startMs, endMs }) => ({
      data: data.runs
        .map((item) => item.trace.summary)
        .filter((trace) => Date.parse(trace.start_time) >= startMs && Date.parse(trace.start_time) <= endMs),
      next_cursor: null,
    }),
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
