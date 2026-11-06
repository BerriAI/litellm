"use client";

import { useState } from "react";
import { ChevronRight, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";
import type { PinnedWorkflowDefinition, RunSteps, StepTrace } from "./contracts";
import { formatCost, formatTimestamp, isActive, stepDescription, stepDuration } from "./presentation";
import { RunStatusBadge } from "./RunStatusBadge";
import { TracePayload } from "./TracePayload";
import { WorkflowStepSpendDetails } from "./WorkflowStepSpendDetails";

export interface WorkflowRunInspectorProps {
  readonly trace: RunSteps;
  readonly definition: PinnedWorkflowDefinition | null;
  readonly selectedStepId: string | null;
  readonly onSelectStep: (stepId: string) => void;
}

function StepDetails({ step }: { step: StepTrace }) {
  const [showSpend, setShowSpend] = useState(false);
  const metrics = [
    ["Duration", stepDuration(step)],
    ["Attempts", step.attempts.toString()],
    ["Input tokens", step.prompt_tokens.toLocaleString()],
    ["Output tokens", step.completion_tokens.toLocaleString()],
    ["Cost", formatCost(step.response_cost)],
  ];
  return (
    <section aria-label="Step details" className="min-w-0 space-y-5 p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-semibold">{step.node_id}</h3>
          <p className="mt-1 break-all font-mono text-xs text-muted-foreground">{step.step_id}</p>
        </div>
        <RunStatusBadge state={step.state} />
      </div>
      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3 xl:grid-cols-5">
        {metrics.map(([label, value]) => (
          <div key={label}>
            <dt className="text-xs text-muted-foreground">{label}</dt>
            <dd className="mt-1 text-sm font-medium tabular-nums">{value}</dd>
          </div>
        ))}
      </dl>
      <dl className="grid gap-2 text-xs text-muted-foreground sm:grid-cols-2">
        <div>
          <dt className="inline">Started </dt>
          <dd className="inline">{formatTimestamp(step.started_at)}</dd>
        </div>
        <div>
          <dt className="inline">Finished </dt>
          <dd className="inline">{step.finished_at ? formatTimestamp(step.finished_at) : "Not finished"}</dd>
        </div>
        <div>
          <dt className="inline">Deadline </dt>
          <dd className="inline">{formatTimestamp(step.deadline)}</dd>
        </div>
      </dl>
      {step.error !== null && (
        <div role="alert" className="rounded-lg border border-destructive/30 bg-destructive/5 p-4">
          <h4 className="text-sm font-medium text-destructive">Step error</h4>
          <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words text-xs">{step.error}</pre>
        </div>
      )}
      <div className="grid min-w-0 gap-4 lg:grid-cols-2">
        <TracePayload label="Input" value={step.input} />
        <TracePayload label="Output" value={step.output} />
      </div>
      <div className="border-t pt-4">
        {step.request_id === null ? (
          <p className="text-xs text-muted-foreground">No spend-log request ID was reported for this step.</p>
        ) : (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="break-all font-mono text-xs text-muted-foreground">{step.request_id}</p>
              <Button variant="outline" size="sm" aria-expanded={showSpend} onClick={() => setShowSpend(!showSpend)}>
                <FileText />
                {showSpend ? "Hide spend record" : "View spend record"}
              </Button>
            </div>
            {showSpend && <WorkflowStepSpendDetails requestId={step.request_id} startedAt={step.started_at} />}
          </div>
        )}
      </div>
    </section>
  );
}

export function WorkflowRunInspector({ trace, definition, selectedStepId, onSelectStep }: WorkflowRunInspectorProps) {
  const selectedStep = trace.steps.find((step) => step.step_id === selectedStepId);
  const reportedCosts = trace.steps.flatMap((step) => (step.response_cost === null ? [] : [step.response_cost]));
  const reportedCost = reportedCosts.length ? reportedCosts.reduce((total, cost) => total + cost, 0) : null;
  const costsIncomplete = trace.steps.some((step) => step.response_cost === null);
  return (
    <section aria-label="Run inspector" className="min-w-0 overflow-hidden rounded-xl border bg-card">
      <div className="space-y-3 border-b px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-base font-semibold">Run inspector</h2>
          <div className="flex items-center gap-3">
            {isActive(trace.state) && <span className="text-xs text-muted-foreground">Updates every 2 seconds</span>}
            <RunStatusBadge state={trace.state} />
          </div>
        </div>
        <dl className="flex flex-wrap gap-x-8 gap-y-3 text-xs">
          <div>
            <dt className="text-muted-foreground">Run ID</dt>
            <dd className="mt-1 break-all font-mono">{trace.run_id}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Pinned snapshot</dt>
            <dd className="mt-1 break-all font-mono">{trace.snapshot_id}</dd>
          </div>
          <div>
            <dt className="text-muted-foreground">Reported step cost</dt>
            <dd className="mt-1 font-medium">
              {formatCost(reportedCost)}
              {costsIncomplete && " · Partial"}
            </dd>
          </div>
        </dl>
        {trace.cancel_requested && (
          <p className="text-xs text-muted-foreground">
            Cancellation requested. Waiting for the gateway to report the final state.
          </p>
        )}
      </div>
      <div className="grid min-w-0 lg:grid-cols-[15rem_minmax(0,1fr)]">
        <nav aria-label="Run steps" className="border-b bg-muted/20 p-3 lg:border-r lg:border-b-0">
          <p className="px-2 pb-3 pt-1 text-xs font-medium text-muted-foreground">
            {trace.steps.length} reported steps
          </p>
          <ol className="space-y-1">
            {trace.steps.map((step, index) => (
              <li key={step.step_id}>
                <Button
                  variant="ghost"
                  aria-pressed={step.step_id === selectedStepId}
                  aria-label={`Inspect step ${index + 1} ${step.node_id}`}
                  onClick={() => onSelectStep(step.step_id)}
                  className={cn(
                    "h-auto w-full justify-start gap-3 whitespace-normal px-3 py-3 text-left",
                    step.step_id === selectedStepId && "bg-background shadow-sm ring-1 ring-border",
                  )}
                >
                  <span className="text-xs text-muted-foreground">{index + 1}</span>
                  <span className="min-w-0 flex-1 space-y-1.5">
                    <span className="block truncate text-sm font-medium">{step.node_id}</span>
                    <span className="block truncate text-xs text-muted-foreground">
                      {stepDescription(definition, step.node_id)}
                    </span>
                    <RunStatusBadge state={step.state} />
                  </span>
                  <ChevronRight className="text-muted-foreground" />
                </Button>
              </li>
            ))}
          </ol>
        </nav>
        {selectedStep ? (
          <StepDetails key={selectedStep.step_id} step={selectedStep} />
        ) : (
          <p className="p-8 text-sm text-muted-foreground">
            {trace.steps.length ? "Select a step to inspect its details." : "No steps have been reported yet."}
          </p>
        )}
      </div>
    </section>
  );
}
