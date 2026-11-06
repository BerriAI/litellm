"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import type {
  PinnedWorkflowDefinition,
  RunSteps,
  WorkflowRunDataSource,
  WorkflowRunGraphState,
  WorkflowStepSelection,
} from "./contracts";
import { isActive, stepSelection } from "./presentation";
import { WorkflowRunHistory } from "./WorkflowRunHistory";
import { WorkflowRunInspector } from "./WorkflowRunInspector";

export interface WorkflowRunDebuggerProps {
  readonly workflowId: string;
  readonly dataSource: WorkflowRunDataSource;
  readonly runId: string | null;
  readonly onSelectRun: (runId: string) => void;
  readonly selectedStepId?: string | null;
  readonly onSelectStep?: (selection: WorkflowStepSelection | null) => void;
  readonly onGraphStateChange?: (state: WorkflowRunGraphState | null) => void;
  readonly renderGraph?: (
    state: WorkflowRunGraphState | null,
    selectStep: (stepId: string | null) => void,
  ) => ReactNode;
  readonly pageSize?: number;
}

export function WorkflowRunDebugger(props: WorkflowRunDebuggerProps) {
  return (
    <div className="space-y-5">
      <WorkflowRunHistory
        workflowId={props.workflowId}
        dataSource={props.dataSource}
        selectedRunId={props.runId}
        onSelectRun={props.onSelectRun}
        pageSize={props.pageSize}
      />
      <RunInspection key={`${props.dataSource.cacheKey}:${props.workflowId}:${props.runId}`} {...props} />
    </div>
  );
}

function RunInspection({
  workflowId,
  dataSource,
  runId,
  selectedStepId,
  onSelectStep,
  onGraphStateChange,
  renderGraph,
}: WorkflowRunDebuggerProps) {
  const [localStepId, setLocalStepId] = useState<string | null>();
  const traceOptions: UseQueryOptions<RunSteps> = {
    queryKey: ["workflowSteps", dataSource.cacheKey, workflowId, runId],
    queryFn: ({ signal }) => dataSource.getSteps({ runId: runId!, signal }),
    enabled: runId !== null,
    retry: false,
    refetchInterval: (query) => (query.state.data && isActive(query.state.data.state) ? 2000 : false),
  };
  const traceQuery = useQuery(traceOptions);
  const trace = traceQuery.data;
  const definitionOptions: UseQueryOptions<PinnedWorkflowDefinition> = {
    queryKey: ["workflowRunDefinition", dataSource.cacheKey, workflowId, runId, trace?.snapshot_id],
    queryFn: ({ signal }) => {
      const request = { workflowId, runId: runId!, snapshotId: trace!.snapshot_id, signal };
      return dataSource.getDefinition(request);
    },
    enabled: trace !== undefined,
    retry: false,
    staleTime: Infinity,
  };
  const definitionQuery = useQuery(definitionOptions);
  const selectionId = selectedStepId === undefined ? localStepId : selectedStepId;
  const defaultStep = trace?.steps.find((step) => step.state === "failed") ?? trace?.steps[0];
  const selected = selectionId === undefined ? defaultStep : trace?.steps.find((step) => step.step_id === selectionId);
  const graphState = useMemo<WorkflowRunGraphState | null>(
    () =>
      trace
        ? {
            runId: trace.run_id,
            snapshotId: trace.snapshot_id,
            state: trace.state,
            definition: definitionQuery.data ?? null,
            selection: selected ? stepSelection(trace, selected) : null,
            executions: trace.steps.map((step) => ({ stepId: step.step_id, nodeId: step.node_id, state: step.state })),
          }
        : null,
    [trace, definitionQuery.data, selected],
  );
  useEffect(() => {
    onGraphStateChange?.(graphState);
    return () => onGraphStateChange?.(null);
  }, [graphState, onGraphStateChange]);
  const selectStep = (stepId: string | null) => {
    if (stepId === null) {
      setLocalStepId(null);
      onSelectStep?.(null);
      return;
    }
    const step = trace?.steps.find((candidate) => candidate.step_id === stepId);
    if (!trace || !step) return;
    setLocalStepId(stepId);
    onSelectStep?.(stepSelection(trace, step));
  };
  if (runId === null)
    return (
      <p className="rounded-xl border border-dashed p-10 text-center text-sm text-muted-foreground">
        Select a run to explore its steps and pinned definition.
      </p>
    );
  return (
    <div className="space-y-3">
      {traceQuery.isPending && (
        <p role="status" className="p-5 text-sm text-muted-foreground">
          Loading step trace…
        </p>
      )}
      {traceQuery.isError && (
        <div role="alert" className="rounded-lg border p-4 text-sm">
          Could not refresh this run. Any displayed steps are from the last successful response.
          <Button variant="outline" size="sm" className="ml-3" onClick={() => void traceQuery.refetch()}>
            Retry trace
          </Button>
        </div>
      )}
      {trace && definitionQuery.isPending && (
        <p role="status" className="text-xs text-muted-foreground">
          Loading this run&apos;s pinned definition…
        </p>
      )}
      {definitionQuery.isError && (
        <div role="alert" className="rounded-lg border p-4 text-sm">
          Pinned definition unavailable. Step traces are still shown.
          <Button variant="outline" size="sm" className="ml-3" onClick={() => void definitionQuery.refetch()}>
            Retry definition
          </Button>
        </div>
      )}
      {renderGraph?.(graphState, selectStep)}
      {trace && (
        <WorkflowRunInspector
          trace={trace}
          definition={definitionQuery.data ?? null}
          selectedStepId={selected?.step_id ?? null}
          onSelectStep={selectStep}
        />
      )}
    </div>
  );
}
