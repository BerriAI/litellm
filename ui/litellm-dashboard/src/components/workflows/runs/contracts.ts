import { z } from "zod";

export const runStates = ["pending", "running", "completed", "failed", "cancelled"] as const;
export const runStateSchema = z.enum(runStates);
const timestamp = z.iso.datetime({ offset: true });
const identifier = z.string().min(1);

const stepTraceFields = {
  step_id: identifier,
  run_id: identifier,
  node_id: identifier,
  state: runStateSchema,
  input: z.json(),
  output: z.json(),
  attempts: z.number().int().nonnegative(),
  deadline: timestamp,
  started_at: timestamp,
  finished_at: timestamp.nullable(),
  error: z.string().nullable(),
  prompt_tokens: z.number().int().nonnegative(),
  completion_tokens: z.number().int().nonnegative(),
  response_cost: z.number().finite().nonnegative().nullable(),
  request_id: identifier.nullable(),
};
export const stepTraceSchema = z.object(stepTraceFields);

const runStepsFields = {
  run_id: identifier,
  snapshot_id: identifier,
  state: runStateSchema,
  cancel_requested: z.boolean(),
  steps: z.array(stepTraceSchema),
};
export const runStepsSchema = z
  .object(runStepsFields)
  .refine((run) => run.steps.every((step) => step.run_id === run.run_id), "Step belongs to another run")
  .refine((run) => new Set(run.steps.map((step) => step.step_id)).size === run.steps.length, "Duplicate step identity");

export type RunState = z.infer<typeof runStateSchema>;
export type StepTrace = z.infer<typeof stepTraceSchema>;
export type RunSteps = z.infer<typeof runStepsSchema>;
export type JsonValue = z.infer<ReturnType<typeof z.json>>;
export type RunStatusFilter = RunState | "all";

export interface WorkflowRunSummary {
  readonly run_id: string;
  readonly workflow_id: string;
  readonly snapshot_id: string;
  readonly state: RunState;
  readonly created_at: string;
}

export interface WorkflowRunPage {
  readonly runs: readonly WorkflowRunSummary[];
  readonly next_cursor: string | null;
}

export interface PinnedWorkflowDefinition {
  readonly workflow_id: string;
  readonly snapshot_id: string;
  readonly bundle: Readonly<Record<string, JsonValue>>;
}

export interface ListWorkflowRuns {
  readonly workflowId: string;
  readonly cursor: string | null;
  readonly status: RunStatusFilter;
  readonly limit: number;
  readonly signal: AbortSignal;
}

export interface ReadRun {
  readonly runId: string;
  readonly signal: AbortSignal;
}

export interface ReadPinnedDefinition extends ReadRun {
  readonly workflowId: string;
  readonly snapshotId: string;
}

export interface WorkflowRunDataSource {
  readonly cacheKey: string;
  listRuns: (request: ListWorkflowRuns) => Promise<WorkflowRunPage>;
  getSteps: (request: ReadRun) => Promise<RunSteps>;
  getDefinition: (request: ReadPinnedDefinition) => Promise<PinnedWorkflowDefinition>;
}

export interface WorkflowStepSelection {
  readonly runId: string;
  readonly snapshotId: string;
  readonly stepId: string;
  readonly nodeId: string;
}

export interface WorkflowRunGraphState {
  readonly runId: string;
  readonly snapshotId: string;
  readonly state: RunState;
  readonly definition: PinnedWorkflowDefinition | null;
  readonly selection: WorkflowStepSelection | null;
  readonly executions: readonly {
    readonly stepId: string;
    readonly nodeId: string;
    readonly state: RunState;
  }[];
}
