import type { RunState, WorkflowRunGraphState } from "./contracts";

export function runDefinitionKey(state: WorkflowRunGraphState): string {
  return JSON.stringify([state.runId, state.snapshotId]);
}

export function runNodeStates(state: WorkflowRunGraphState): Readonly<Record<string, RunState>> {
  const states: Record<string, RunState> = Object.fromEntries(
    state.executions.map((execution) => [execution.nodeId, execution.state]),
  );
  const selected = state.executions.find((execution) => execution.stepId === state.selection?.stepId);
  return selected ? { ...states, [selected.nodeId]: selected.state } : states;
}

export function stepForNode(state: WorkflowRunGraphState, nodeId: string | null): string | null {
  if (nodeId === null) return null;
  if (state.selection?.nodeId === nodeId) return state.selection.stepId;
  return state.executions.findLast((execution) => execution.nodeId === nodeId)?.stepId ?? null;
}
