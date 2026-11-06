import type {
  PinnedWorkflowDefinition,
  RunSteps,
  StepTrace,
  WorkflowRunDataSource,
  WorkflowRunSummary,
} from "@/components/workflows/runs/contracts";

export const fixtureWorkflowId = "fixture-summary-workflow";

const nodeDefaults = { timeout_seconds: 30, retry: { max_attempts: 2, backoff_seconds: 1 } };
export const fixtureDefinitions: Readonly<Record<string, PinnedWorkflowDefinition>> = {
  "snapshot-fixture-v1": {
    workflow_id: fixtureWorkflowId,
    snapshot_id: "snapshot-fixture-v1",
    bundle: {
      contract_version: "1",
      name: "document-summary",
      timeout_seconds: 90,
      input_schema: { type: "array", items: { type: "object" } },
      output: { from: "node_3", output: "result" },
      nodes: [
        {
          ...nodeDefaults,
          id: "node_1",
          type: "chat_completion",
          model: "summary-model",
          messages: { from: "$input", output: "messages" },
        },
        {
          ...nodeDefaults,
          id: "node_2",
          type: "python",
          inputs: { summary: { from: "node_1", output: "result" } },
          program: {
            contract_version: "1",
            function_name: "format_summary",
            parameter_names: ["summary"],
            source: "def format_summary(summary: str) -> str:\n    return summary.strip()\n",
            input_schema: {
              type: "object",
              properties: { summary: { type: "string" } },
              required: ["summary"],
              additionalProperties: false,
            },
            output_schema: { type: "string" },
            source_sha256: "a7d6ef8657708891bfaa6fb481ef66048bf030877a0197b94e6e0c28859c98573",
          },
        },
        {
          ...nodeDefaults,
          id: "node_3",
          type: "chat_completion",
          model: "review-model",
          messages: { from: "node_2", output: "result" },
        },
      ],
    },
  },
};

const startedAt = "2026-11-06T08:30:00Z";
const stepDefaults = {
  state: "completed",
  input: { messages: [{ role: "user", content: "Summarize the release notes in one sentence." }] },
  output: { summary: "The release improves workflow debugging.", confidence: 0.94 },
  attempts: 1,
  deadline: "2026-11-06T08:31:00Z",
  started_at: startedAt,
  finished_at: "2026-11-06T08:30:02Z",
  error: null,
  prompt_tokens: 142,
  completion_tokens: 36,
  response_cost: 0.00042,
  request_id: "fixture-request-summary",
} satisfies Omit<StepTrace, "step_id" | "node_id" | "run_id">;

export const fixtureTraces: Readonly<Record<string, RunSteps>> = {
  "run-fixture-completed": {
    run_id: "run-fixture-completed",
    snapshot_id: "snapshot-fixture-v1",
    state: "completed",
    cancel_requested: false,
    steps: [
      { ...stepDefaults, run_id: "run-fixture-completed", step_id: "step-completed-1", node_id: "node_1" },
      {
        ...stepDefaults,
        run_id: "run-fixture-completed",
        step_id: "step-completed-2",
        node_id: "node_2",
        input: false,
        output: 0,
        prompt_tokens: 0,
        completion_tokens: 0,
        response_cost: 0,
        request_id: null,
        started_at: "2026-11-06T08:30:02Z",
        finished_at: "2026-11-06T08:30:02.025Z",
      },
      {
        ...stepDefaults,
        run_id: "run-fixture-completed",
        step_id: "step-completed-3",
        node_id: "node_3",
        input: { accepted: true, summary: "The release improves workflow debugging." },
        output: "Ready to publish.",
        request_id: "fixture-request-review",
        started_at: "2026-11-06T08:30:02.025Z",
        finished_at: "2026-11-06T08:30:03.150Z",
      },
    ],
  },
  "run-fixture-running": {
    run_id: "run-fixture-running",
    snapshot_id: "snapshot-fixture-v1",
    state: "running",
    cancel_requested: false,
    steps: [
      { ...stepDefaults, run_id: "run-fixture-running", step_id: "step-running-1", node_id: "node_1" },
      {
        ...stepDefaults,
        run_id: "run-fixture-running",
        step_id: "step-running-2",
        node_id: "node_2",
        state: "running",
        input: "",
        output: null,
        prompt_tokens: 0,
        completion_tokens: 0,
        response_cost: null,
        request_id: null,
        finished_at: null,
      },
    ],
  },
  "run-fixture-failed": {
    run_id: "run-fixture-failed",
    snapshot_id: "snapshot-fixture-v1",
    state: "failed",
    cancel_requested: false,
    steps: [
      { ...stepDefaults, run_id: "run-fixture-failed", step_id: "step-failed-1", node_id: "node_1" },
      {
        ...stepDefaults,
        run_id: "run-fixture-failed",
        step_id: "step-failed-2",
        node_id: "node_2",
        state: "failed",
        input: { summary: null },
        output: null,
        attempts: 2,
        error: "ValidationError: summary must be a string. Received null.",
        prompt_tokens: 0,
        completion_tokens: 0,
        response_cost: null,
        request_id: null,
      },
    ],
  },
  "run-fixture-repeated-node": {
    run_id: "run-fixture-repeated-node",
    snapshot_id: "snapshot-fixture-v1",
    state: "completed",
    cancel_requested: false,
    steps: [
      {
        ...stepDefaults,
        run_id: "run-fixture-repeated-node",
        step_id: "step-repeat-a",
        node_id: "node_1",
        output: "First execution",
      },
      {
        ...stepDefaults,
        run_id: "run-fixture-repeated-node",
        step_id: "step-repeat-b",
        node_id: "node_1",
        output: "Second execution",
        request_id: "fixture-request-second",
      },
    ],
  },
  "run-fixture-pending": {
    run_id: "run-fixture-pending",
    snapshot_id: "snapshot-fixture-v1",
    state: "pending",
    cancel_requested: false,
    steps: [],
  },
};

export const fixtureRunSummaries: readonly WorkflowRunSummary[] = Object.values(fixtureTraces).map((trace) => ({
  run_id: trace.run_id,
  workflow_id: fixtureWorkflowId,
  snapshot_id: trace.snapshot_id,
  state: trace.state,
  created_at: startedAt,
}));

export function createWorkflowRunFixtureSource(): WorkflowRunDataSource {
  return {
    cacheKey: crypto.randomUUID(),
    async listRuns({ workflowId, status, cursor, limit, signal }) {
      signal.throwIfAborted();
      const matching = fixtureRunSummaries.filter(
        (run) => run.workflow_id === workflowId && (status === "all" || run.state === status),
      );
      const start = cursor ? matching.findIndex((run) => run.run_id === cursor) + 1 : 0;
      const runs = matching.slice(start, start + limit);
      return { runs, next_cursor: start + limit < matching.length ? runs[runs.length - 1].run_id : null };
    },
    async getSteps({ runId, signal }) {
      signal.throwIfAborted();
      const trace = fixtureTraces[runId];
      if (!trace) throw new Error("Fixture run not found");
      return trace;
    },
    async getDefinition({ workflowId, snapshotId, signal }) {
      signal.throwIfAborted();
      const definition = fixtureDefinitions[snapshotId];
      if (!definition || definition.workflow_id !== workflowId) throw new Error("Fixture snapshot not found");
      return definition;
    },
  };
}
