"use client";

import { useMemo, useState } from "react";
import {
  createWorkflowRunApi,
  WorkflowRunDebugger,
  WorkflowRunGraphPanel,
  type WorkflowRunGraphState,
} from "@/components/workflows/runs";
import { useAuth } from "@/contexts/AuthContext";
import { fixtureWorkflowId } from "./data";

function renderGraph(state: WorkflowRunGraphState | null, onSelectStep: (stepId: string | null) => void) {
  return <WorkflowRunGraphPanel state={state} onSelectStep={onSelectStep} />;
}

export default function WorkflowRunDemo() {
  const { accessToken } = useAuth();
  const source = useMemo(
    () => createWorkflowRunApi({ accessToken: accessToken ?? "", teamId: "fixture-team" }),
    [accessToken],
  );
  const [runId, setRunId] = useState<string | null>(null);
  const [graphState, setGraphState] = useState<WorkflowRunGraphState | null>(null);
  return (
    <div className="mx-auto max-w-7xl space-y-5 p-6">
      <div>
        <p className="mb-2 text-xs text-muted-foreground">Workflows / Execution history</p>
        <h1 className="text-2xl font-semibold tracking-tight">Document summary</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Inspect runs, follow each step, and debug its exact inputs and outputs
        </p>
      </div>
      <div className="rounded-lg border border-info/30 bg-info/5 px-4 py-3 text-sm">
        <strong>Fixture preview</strong> · Synthetic traces and spend records. No workflow execution or live billing.
      </div>
      <WorkflowRunDebugger
        workflowId={fixtureWorkflowId}
        dataSource={source}
        runId={runId}
        onSelectRun={setRunId}
        onGraphStateChange={setGraphState}
        renderGraph={renderGraph}
        pageSize={3}
      />
      {graphState && (
        <p className="break-all text-xs text-muted-foreground">
          Graph selection · {graphState.selection?.nodeId ?? "None"} ·{" "}
          {graphState.selection?.stepId ?? "No step selected"} · {graphState.snapshotId}
        </p>
      )}
    </div>
  );
}
