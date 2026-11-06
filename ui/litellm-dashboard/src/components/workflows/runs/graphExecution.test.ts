import { describe, expect, it } from "vitest";
import type { WorkflowRunGraphState } from "./contracts";
import { runDefinitionKey, runNodeStates, stepForNode } from "./graphExecution";

const state: WorkflowRunGraphState = {
  runId: "run-one",
  snapshotId: "snapshot-one",
  state: "running",
  definition: null,
  selection: null,
  executions: [
    { stepId: "execution-a", nodeId: "node_1", state: "failed" },
    { stepId: "execution-b", nodeId: "node_1", state: "running" },
  ],
};

describe("run graph selection", () => {
  it("defaults to the last reported execution while preserving the selected execution's state", () => {
    expect(runNodeStates(state)).toEqual({ node_1: "running" });
    expect(stepForNode(state, "node_1")).toBe("execution-b");
    const selected = {
      ...state,
      selection: { runId: state.runId, snapshotId: state.snapshotId, stepId: "execution-a", nodeId: "node_1" },
    };
    expect(runNodeStates(selected)).toEqual({ node_1: "failed" });
    expect(stepForNode(selected, "node_1")).toBe("execution-a");
    expect(selected.executions).toHaveLength(2);
  });

  it("does not invent executions or states for unreported nodes", () => {
    expect(stepForNode(state, "node_2")).toBeNull();
    expect(stepForNode(state, null)).toBeNull();
    expect(runNodeStates(state)).not.toHaveProperty("node_2");
  });

  it("isolates execution overlays by both run and pinned snapshot", () => {
    expect(runDefinitionKey({ ...state, runId: "another-run" })).not.toBe(runDefinitionKey(state));
    expect(runDefinitionKey({ ...state, snapshotId: "another-snapshot" })).not.toBe(runDefinitionKey(state));
  });
});
