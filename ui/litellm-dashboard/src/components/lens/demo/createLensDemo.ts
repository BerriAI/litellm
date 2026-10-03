import { ApiError } from "@/lib/http/client";
import type { TracesApi } from "@/components/view_logs/TraceView/tracesApi";
import type { LensDemo } from "../LensDemoContext";
import type { LensServices } from "../services";
import type { LensApi } from "../api/service";
import type { Trace, Span, SpanDetail } from "@/components/view_logs/TraceView/traceTypes";
import type { Lens, Finding, Job, Settings } from "../model/types";
import { withReleaseCases } from "./lensDemoLongTrace";

import { scenarios, type Scenario } from "./scenarios";

const executionId = (traceId: string) => btoa(JSON.stringify(["traces", "", traceId]));
const iso = (time: number) => new Date(time).toISOString();

function makeTrace(scene: Scenario, index: number, now: number) {
  const traceId = (index + 1).toString(16).padStart(32, "0");
  const spanId = (step: number) => `${index + 1}${step}`.padStart(16, "0");
  const toolCount = scene.failed && index !== 9 ? 3 : 1;
  const duration = toolCount === 3 ? 6340 : 3180 + index * 137;
  const base: Span = {
    agent: scene.agent,
    duration_ms: duration,
    error: null,
    error_truncated: false,
    framework: "",
    input_preview: scene.question,
    input_tokens: 0,
    output_tokens: 0,
    litellm_request_id: null,
    model: null,
    name: scene.agent,
    parent_span_id: null,
    span_id: spanId(0),
    spend: null,
    start_offset_ms: 0,
    status: "ok",
    type: "agent",
  };
  const tools: Span[] = Array.from({ length: toolCount }, (_, i) => ({
    ...base,
    span_id: spanId(i + 1),
    parent_span_id: base.span_id,
    name: scene.tool,
    type: "tool",
    start_offset_ms: 150 + i * 1600,
    duration_ms: scene.failed ? 1500 : 340,
    status: scene.failed ? "error" : "ok",
    error: scene.failed ? scene.result : null,
  }));
  const model: Span = {
    ...base,
    span_id: spanId(5),
    parent_span_id: base.span_id,
    name: "Generate response",
    type: "llm",
    model: "demo-chat-model",
    start_offset_ms: toolCount * 1600,
    duration_ms: 1200,
    input_tokens: 520 + index * 41,
    output_tokens: 48 + index * 7,
    spend: 0.003 + index * 0.0002,
  };
  const trace: Trace = {
    summary: {
      agent_count: 1,
      agent_invocations: 1,
      agent_names: [scene.agent],
      duration_ms: duration,
      error_count: scene.failed ? toolCount : 0,
      frameworks: [],
      input_preview: scene.question,
      input_tokens: model.input_tokens,
      output_tokens: model.output_tokens,
      llm_calls: 1,
      models: [model.model!],
      name: scene.agent,
      service: "demo-agents",
      span_count: toolCount + 2,
      spend: model.spend,
      start_time: iso(now - (index + 1) * 35 * 60_000),
      status: scene.failed ? "error" : "ok",
      tool_calls: toolCount,
      trace_id: traceId,
    },
    agents: [
      {
        name: scene.agent,
        parent_agent: null,
        duration_ms: duration,
        invocations: 1,
        llm_calls: 1,
        tool_calls: toolCount,
        spend: model.spend,
      },
    ],
    spans: [base, ...tools, model],
  };
  const details: SpanDetail[] = trace.spans.map((span) => ({
    span_id: span.span_id,
    input:
      span.type === "tool"
        ? JSON.stringify({ query: scene.question })
        : JSON.stringify([{ role: "user", content: scene.question }]),
    output: span.type === "tool" ? scene.result : JSON.stringify([{ role: "assistant", content: scene.answer }]),
    attributes: {
      "gen_ai.agent.name": scene.agent,
      "service.name": "demo-agents",
      demo: "true",
      ...(span.model ? { "gen_ai.request.model": span.model } : {}),
    },
  }));
  return { trace, details };
}

export type LensDemoData = ReturnType<typeof createLensDemoData>;

export function createLensDemoData(now = Date.now()) {
  const runs = scenarios.map((scene, index) => {
    const run = makeTrace(scene, index, now);
    return index === 6 ? withReleaseCases(run) : run;
  });
  const finding = ({
    id,
    check,
    title,
    description,
    suggestion,
    indices,
    kind = "issue",
  }: {
    id: string;
    check: string;
    title: string;
    description: string;
    suggestion: string;
    indices: number[];
    kind?: Finding["kind"];
  }): Finding => ({
    id,
    check_id: check,
    title,
    description,
    suggestion,
    kind,
    priority: kind === "issue" ? "high" : "low",
    status: "open",
    reason: "",
    revision: 1,
    first_seen: iso(now - 86_400_000),
    last_seen: iso(now - 300_000),
    limitation: "",
    occurrences: indices.map((i) => executionId(runs[i].trace.summary.trace_id)),
    evidence: indices.map((i) => ({
      execution_id: executionId(runs[i].trace.summary.trace_id),
      span_id: runs[i].trace.spans.at(-1)!.span_id,
      quote: scenarios[i].answer,
      role: "support",
    })),
  });
  const findingInputs: Parameters<typeof finding>[0][] = [
    {
      id: "failed-lookups",
      check: "recover",
      title: "Repeated lookups leave customers without an answer",
      description:
        "The support agent retries the same unavailable order service three times, then promises to check without answering or offering a handoff.",
      suggestion: "After repeated failures, explain the problem and offer a handoff.",
      indices: [0, 4],
    },
    {
      id: "safe-handoff",
      check: "recover",
      title: "A clear handoff helps when the order service is unavailable",
      description:
        "The agent explains the service outage and offers a support handoff instead of promising an answer it cannot provide.",
      suggestion: "Keep this fallback for unavailable services.",
      indices: [9],
      kind: "pattern",
    },
    {
      id: "unsupported-claim",
      check: "grounding",
      title: "Performance claim has no supporting benchmark",
      description:
        "The answer claims a 40% performance improvement, but the retrieved documentation contains no comparative benchmark.",
      suggestion: "Require a benchmark source for numeric performance claims, or remove the comparison.",
      indices: [1],
    },
    {
      id: "uncertainty",
      check: "grounding",
      title: "Missing information is acknowledged",
      description:
        "When documentation does not establish regional availability, the agent says so and asks the user to verify it.",
      suggestion: "Keep stating when a source does not answer the question.",
      indices: [8],
      kind: "pattern",
    },
    {
      id: "release-blocked",
      check: "release",
      title: "Failing checks correctly block the release",
      description: "The release agent identifies an unresolved regression and recommends holding the release.",
      suggestion: "Continue requiring passing checks before recommending a release.",
      indices: [6],
      kind: "pattern",
    },
  ];
  const findings = findingInputs.map(finding);
  const definitions = [
    {
      id: "support",
      name: "Support quality",
      agent: "support_agent",
      check: "recover",
      context:
        "Answer the customer's question using order information. If a tool fails, explain the problem and offer a handoff.",
      instruction: "Look for repeated failed calls and conversations that end without an answer or a handoff.",
    },
    {
      id: "research",
      name: "Research accuracy",
      agent: "research_agent",
      check: "grounding",
      context: "Answer questions using verified documentation. Acknowledge missing information.",
      instruction: "Find claims that are not supported by the retrieved sources.",
    },
    {
      id: "release",
      name: "Release readiness",
      agent: "release_agent",
      check: "release",
      context: "Review test results and recommend a release only when all required checks pass.",
      instruction: "Check whether failed tests are acknowledged before a release recommendation.",
    },
  ];
  const lenses: Lens[] = definitions.map((definition) => {
    const settings: Settings = {
      name: definition.name,
      agent_name: definition.agent,
      context: definition.context,
      checks: [{ id: definition.check, instruction: definition.instruction, enabled: true }],
      source: "traces",
      service: "",
      filters: [],
      team_id: "",
      execution_ids: [],
      lookback_hours: 24,
      sample_size: 0,
      sample_percent: 100,
      concurrency: 8,
      monthly_budget: 100,
      interval_minutes: 30,
      enabled: false,
      model: "demo-analysis-model",
    };
    const executions = runs
      .filter(({ trace }) => trace.summary.name === definition.agent)
      .map(({ trace }) => ({
        id: executionId(trace.summary.trace_id),
        trace_ref: "",
        metadata: [],
        root_seen: true,
        service: trace.summary.service,
        source: "traces" as const,
        trace_id: trace.summary.trace_id,
        team_id: "",
        name: trace.summary.name,
        start_time: trace.summary.start_time,
        span_count: trace.summary.span_count,
      }));
    const relevant = findings.filter((f) => f.check_id === definition.check);
    const jobs: Job[] = [0, 1].map((day) => {
      const sample = day === 0 ? executions : executions.slice(1);
      const selectedIds = new Set(sample.map((item) => item.id));
      const snapshot = relevant
        .map((f) => ({
          ...f,
          occurrences: f.occurrences.filter((id) => selectedIds.has(id)),
          evidence: f.evidence.filter((e) => selectedIds.has(e.execution_id)),
        }))
        .filter((f) => f.occurrences.length > 0);
      return {
        id: `${definition.id}-scan-${day}`,
        findings: snapshot,
        settings,
        revision: 1,
        assessments: sample.map((e) => ({
          execution_id: e.id,
          cannot_assess: false,
          issue_checks: snapshot.some((f) => f.kind === "issue" && f.occurrences.includes(e.id))
            ? [definition.check]
            : [],
          pattern_checks: snapshot.some((f) => f.kind === "pattern" && f.occurrences.includes(e.id))
            ? [definition.check]
            : [],
        })),
        attempts: 1,
        error: "",
        cost: sample.length * 0.012,
        coverage: {
          eligible: sample.length,
          selected: sample.length,
          screened: sample.length,
          investigated: sample.length,
          inconclusive: 0,
          grouping_batches: 1,
          grouped_batches: 1,
          candidates: snapshot.length,
          partial: 0,
          unassessable: 0,
        },
        status: "completed",
        stage: "Complete",
        created_at: iso(now - 330_000 - day * 60_000),
        finished_at: iso(now - 300_000 - day * 60_000),
        start: iso(now - 86_400_000),
        end: iso(now - 300_000 - day * 60_000),
        sample: { eligible: sample.length, selected: sample.length, executions: sample },
      };
    });
    return {
      id: definition.id,
      version: 1,
      revision: 1,
      spent: jobs.reduce((sum, job) => sum + job.cost, 0),
      scope: { all_teams: true, api_key_hash: "", team_id: "" },
      settings,
      created_at: iso(now - 2 * 86_400_000),
      next_run_at: iso(now),
      budget_month: iso(now).slice(0, 7),
      findings: relevant,
      jobs,
    };
  });
  return { runs, lenses };
}

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

export interface LensDemoSession extends LensDemo {
  readonly services: LensServices;
}

export function createLensDemo(now = Date.now()): LensDemoSession {
  const data = createLensDemoData(now);
  return {
    services: { lens: demoLensApi(data), traces: demoTracesApi(data) },
    copyTrace: (traceId, spanId) => {
      const run = data.runs.find(({ trace }) => trace.summary.trace_id === traceId);
      return JSON.stringify(spanId ? run?.details.find((span) => span.span_id === spanId) : run, null, 2);
    },
  };
}
