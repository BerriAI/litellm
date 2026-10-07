import { ApiError } from "@/lib/http/client";
import type { TracesApi } from "@/components/lens/traces/api";
import { rollUpAgents } from "@/components/lens/agents/agentRollup";
import type { LensServices } from "../LensServices";
import type { LensApi } from "../service";
import { demoDatasetsApi } from "./demoDatasets";
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
    datasets: demoDatasetsApi(data),
    lenses: async () => ({ lenses: data.lenses, workers: [], tracing_enabled: true }),
    activity: async () => ({ traces: true, requests: false }),
    runs: (lensId, offset) => found(jobs(lensId)?.slice(offset)),
    run: (lensId, jobId) => found(jobs(lensId)?.find((job) => job.id === jobId)),
    reviews: async (lensId, jobId, after) => {
      const job = await found(jobs(lensId)?.find((item) => item.id === jobId));
      return {
        reviews: job.reviews.slice(Math.max(0, after - (job.reviewed - job.reviews.length))),
        reviewed: job.reviewed,
      };
    },
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
    signalConfig: async () => ({ model: "", threshold: 0.5, signals: [] }),
    saveSignalConfig: readOnly,
    cancelRun: readOnly,
    reviewFinding: readOnly,
    registerWorker: readOnly,
    setWorkerBillingKey: readOnly,
    revokeWorker: readOnly,
    generateAnalysisKey: readOnly,
    deleteKeys: readOnly,
  };
}

const summariesIn = (data: LensDemoData, startMs: number, endMs: number) =>
  data.runs
    .map((item) => item.trace.summary)
    .filter((trace) => Date.parse(trace.start_time) >= startMs && Date.parse(trace.start_time) <= endMs);

function demoTracesApi(data: LensDemoData): TracesApi {
  const run = (traceId: string) => data.runs.find(({ trace }) => trace.summary.trace_id === traceId);
  return {
    live: false,
    handoff: (traceId, spanId) => {
      const found = run(traceId);
      const step = spanId ? found?.details.find((span) => span.span_id === spanId) : found;
      return { text: JSON.stringify(step, null, 2), copied: spanId ? "Step copied" : "Trace copied" };
    },
    list: async ({ startMs, endMs }) => ({ data: summariesIn(data, startMs, endMs), next_cursor: null }),
    agents: async ({ startMs, endMs }) => rollUpAgents(summariesIn(data, startMs, endMs)),
    findings: async (traces) =>
      traces.map((trace) => {
        const jobs = data.lenses.flatMap((lens) => lens.jobs).filter((job) => job.status === "completed");
        const assessed = jobs.flatMap((job) =>
          (job.sample?.executions ?? [])
            .filter((execution) => {
              const matches =
                execution.source === "traces" &&
                execution.trace_id === trace.trace_id &&
                (execution.trace_ref ?? "") === (trace.trace_ref ?? "");
              return (
                matches &&
                job.assessments.some(
                  (assessment) => assessment.execution_id === execution.id && !assessment.cannot_assess,
                )
              );
            })
            .map((execution) => ({ job, execution })),
        );
        const findings = new Set(
          assessed.flatMap(({ job, execution }) =>
            (job.findings ?? [])
              .filter((finding) => finding.occurrences.includes(execution.id))
              .map((finding) => finding.id),
          ),
        );
        return { ...trace, finding_count: assessed.length ? findings.size : null };
      }),
    signals: async (traces) =>
      traces.map((trace) => ({ ...trace, status: "unclassified" as const, flags: [], model: "", classified_at: null })),
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
