"use client";

import { useState } from "react";
import { GitBranch } from "lucide-react";
import { Button } from "@/components/ui/button";
import { WorkflowDefinitionView, type WorkflowExecutionOverlay } from "@/components/workflows";
import draft from "./draft";

const definitionKey = `${draft.workflow_id}:${draft.etag}`;
const examples = ["Definition", "Running", "Failed", "Completed", "Loading", "Empty", "Unsupported"] as const;
type Example = (typeof examples)[number];

function overlayFor(example: Example): WorkflowExecutionOverlay | undefined {
  if (example === "Running")
    return { definitionKey, states: { node_1: "completed", node_2: "running", node_3: "pending" } };
  if (example === "Failed")
    return { definitionKey, states: { node_1: "completed", node_2: "failed", node_3: "cancelled" } };
  if (example === "Completed")
    return { definitionKey, states: { node_1: "completed", node_2: "completed", node_3: "completed" } };
  return undefined;
}

function bundleFor(example: Example) {
  if (example === "Empty") return null;
  if (example === "Unsupported") return { ...draft.bundle, nodes: [{ id: "future_node", type: "if" }] };
  return draft.bundle;
}

export default function WorkflowGraphDemo() {
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [example, setExample] = useState<Example>("Definition");
  return (
    <div className="mx-auto max-w-7xl space-y-5 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="space-y-1">
          <p className="flex items-center gap-2 text-xs text-muted-foreground">
            <GitBranch className="size-3.5" />
            Workflows / Definition preview
          </p>
          <h1 className="text-2xl font-semibold tracking-tight">{draft.name}</h1>
          <p className="text-sm text-muted-foreground">Inspect how each step transforms your input</p>
        </div>
        <span className="rounded-full border border-border bg-muted px-3 py-1 text-xs font-medium">
          Read-only definition
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-muted/40 p-3">
        <span className="mr-2 text-xs font-medium text-muted-foreground">Fixture preview</span>
        {examples.map((name) => (
          <Button
            key={name}
            size="sm"
            variant={example === name ? "default" : "ghost"}
            aria-pressed={example === name}
            onClick={() => setExample(name)}
          >
            {name}
          </Button>
        ))}
      </div>
      <WorkflowDefinitionView
        bundle={bundleFor(example)}
        definitionKey={definitionKey}
        selectedNodeId={selectedNodeId}
        onNodeSelect={setSelectedNodeId}
        execution={overlayFor(example)}
        loading={example === "Loading"}
      />
      <p className="text-xs text-muted-foreground">
        Development only. Definitions compiled by the LiteFlow PR #3 SDK. Execution states are simulated fixtures. No
        gateway calls or workflow execution.
      </p>
    </div>
  );
}
