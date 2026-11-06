"use client";

import { useState } from "react";
import { ChevronDown, GitBranch } from "lucide-react";
import { WorkflowGraph } from "@/components/workflows";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { WorkflowRunGraphState } from "./contracts";
import { runDefinitionKey, runNodeStates, stepForNode } from "./graphExecution";

export interface WorkflowRunGraphPanelProps {
  readonly state: WorkflowRunGraphState | null;
  readonly onSelectStep: (stepId: string | null) => void;
}

export function WorkflowRunGraphPanel({ state, onSelectStep }: WorkflowRunGraphPanelProps) {
  if (state === null) return null;
  return <PinnedRunGraph key={runDefinitionKey(state)} state={state} onSelectStep={onSelectStep} />;
}

function PinnedRunGraph({ state, onSelectStep }: WorkflowRunGraphPanelProps & { state: WorkflowRunGraphState }) {
  const [unreportedNode, setUnreportedNode] = useState<string | null>(null);
  const definitionKey = runDefinitionKey(state);
  const selectNode = (nodeId: string | null) => {
    const stepId = stepForNode(state, nodeId);
    setUnreportedNode(stepId === null ? nodeId : null);
    onSelectStep(stepId);
  };
  return (
    <Collapsible className="rounded-xl border bg-card">
      <CollapsibleTrigger render={<Button variant="ghost" className="h-auto w-full justify-between px-5 py-4" />}>
        <span className="flex items-center gap-2">
          <GitBranch />
          Pinned workflow
        </span>
        <ChevronDown />
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-3 px-4 pb-4">
        <p className="break-all text-xs text-muted-foreground">
          Snapshot {state.snapshotId}. Select a node to inspect its reported execution.
        </p>
        {state.definition ? (
          <WorkflowGraph
            bundle={state.definition.bundle}
            definitionKey={definitionKey}
            selectedNodeId={state.selection?.nodeId ?? unreportedNode}
            onNodeSelect={selectNode}
            execution={{ definitionKey, states: runNodeStates(state) }}
          />
        ) : (
          <p className="p-4 text-sm text-muted-foreground">
            The pinned definition has not loaded. The current draft is never substituted.
          </p>
        )}
        {unreportedNode && !state.selection && (
          <p role="status" className="text-sm text-muted-foreground">
            No step has been reported for {unreportedNode}.
          </p>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
