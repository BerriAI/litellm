"use client";

import { cn } from "@/lib/cva.config";

import { stepLine } from "../model/inbox";
import type { Job } from "../model/types";

const STEP_MARK = { model: "✓", stage: "▸", error: "!" } as const;
const STEP_TONE = {
  model: "text-foreground",
  stage: "text-info",
  error: "text-destructive",
} as const;

export function StepFeed({ job }: { job: Job }) {
  const steps = job.steps ?? [];
  return (
    <ol
      aria-label="Investigation steps"
      className="max-h-56 space-y-0.5 overflow-y-auto font-mono text-xs"
      data-testid="step-feed"
    >
      {steps.length === 0 && <li className="text-muted-foreground">Waiting for the worker to start…</li>}
      {[...steps].reverse().map((step, index) => (
        <li
          key={`${step.at}-${index}`}
          className="grid grid-cols-[1rem_4.5rem_minmax(0,1fr)_auto] items-baseline gap-2"
        >
          <span aria-hidden="true" className={STEP_TONE[step.kind]}>
            {STEP_MARK[step.kind]}
          </span>
          <span className="tabular-nums text-muted-foreground">
            {new Date(step.at).toLocaleTimeString(undefined, { hour12: false })}
          </span>
          <span className={cn("truncate", STEP_TONE[step.kind])}>{stepLine(step)}</span>
          <span className="truncate text-muted-foreground">{step.model}</span>
        </li>
      ))}
    </ol>
  );
}
