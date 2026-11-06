import type { PinnedWorkflowDefinition, RunState, RunSteps, StepTrace, WorkflowStepSelection } from "./contracts";

export const stateLabels: Record<RunState, string> = {
  pending: "Pending",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  cancelled: "Cancelled",
};

export function isActive(state: RunState): boolean {
  return state === "pending" || state === "running";
}

export function stepDuration(step: StepTrace): string {
  if (step.finished_at === null) return isActive(step.state) ? "In progress" : "Not reported";
  const milliseconds = Date.parse(step.finished_at) - Date.parse(step.started_at);
  if (milliseconds < 0 || !Number.isFinite(milliseconds)) return "Not reported";
  return milliseconds < 1000 ? `${milliseconds} ms` : `${(milliseconds / 1000).toFixed(2)} s`;
}

export function formatCost(cost: number | null): string {
  return cost === null ? "Not reported" : `$${cost.toFixed(6)}`;
}

export function formatTimestamp(timestamp: string): string {
  const options: Intl.DateTimeFormatOptions = {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZoneName: "short",
  };
  return new Date(timestamp).toLocaleString(undefined, options);
}

export function stepSelection(trace: RunSteps, step: StepTrace): WorkflowStepSelection {
  return { runId: trace.run_id, snapshotId: trace.snapshot_id, stepId: step.step_id, nodeId: step.node_id };
}

export function stepDescription(definition: PinnedWorkflowDefinition | null, nodeId: string): string {
  const nodes = definition?.bundle.nodes;
  if (!Array.isArray(nodes)) return "Workflow step";
  const node = nodes.find((entry) => {
    if (entry === null || typeof entry !== "object" || Array.isArray(entry)) return false;
    return entry.id === nodeId;
  });
  if (node === null || typeof node !== "object" || Array.isArray(node)) return "Workflow step";
  if (node.type === "python") return "Python";
  if (node.type === "chat_completion") return typeof node.model === "string" ? node.model : "Chat completion";
  return "Workflow step";
}
