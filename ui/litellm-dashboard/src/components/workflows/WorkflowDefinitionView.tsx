"use client";

import { cn } from "@/lib/cva.config";
import { WorkflowGraph, type WorkflowGraphProps } from "./WorkflowGraph";
import { WorkflowDefinitionInspector } from "./WorkflowDefinitionInspector";

export function WorkflowDefinitionView({ className, ...props }: WorkflowGraphProps) {
  return (
    <div className={cn("grid min-w-0 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]", className)}>
      <WorkflowGraph {...props} />
      <WorkflowDefinitionInspector {...props} />
    </div>
  );
}
