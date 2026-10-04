"use client";

import { ChevronDown } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/cva.config";

import { modelsUsed, stepLine, windowLabel } from "../model/inbox";
import type { Job } from "../model/types";

const STEP_MARK = { model: "✓", stage: "▸", error: "!" } as const;
const STEP_TONE = {
  model: "text-foreground",
  stage: "text-info",
  error: "text-destructive",
} as const;

export function StepFeed({ job }: { job: Job }) {
  const [open, setOpen] = useState(true);
  const steps = job.steps ?? [];
  const models = modelsUsed(steps);
  const calls = steps.filter((s) => s.kind === "model").length;
  return (
    <section aria-label="Live investigation steps" className="rounded-md border bg-background font-mono text-xs">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        className="flex w-full flex-wrap items-center justify-between gap-x-4 gap-y-1 px-3 py-2 text-left hover:bg-muted/40"
      >
        <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-muted-foreground">
          <span className="font-semibold text-foreground">
            {job.trigger === "manual" ? "manual run" : "scheduled run"}
          </span>
          <span>{windowLabel(job)}</span>
          <span>
            {calls} model {calls === 1 ? "call" : "calls"} · ${job.cost.toFixed(4)}
          </span>
          {models.length > 0 && (
            <span data-testid="step-feed-models">
              using <span className="text-foreground">{models.join(", ")}</span>
            </span>
          )}
        </span>
        <ChevronDown className={cn("size-3.5 text-muted-foreground transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ol className="max-h-56 space-y-0.5 overflow-y-auto border-t px-3 py-2" data-testid="step-feed">
          {steps.length === 0 && <li className="text-muted-foreground">Waiting for the worker to start…</li>}
          {[...steps].reverse().map((step, index) => (
            <li
              key={`${step.at}-${index}`}
              className={cn(
                "grid grid-cols-[1rem_4.5rem_minmax(0,1fr)_auto] items-baseline gap-2",
                index === 0 && "motion-safe:animate-in motion-safe:fade-in",
              )}
            >
              <span aria-hidden="true" className={STEP_TONE[step.kind]}>
                {index === 0 && step.kind === "stage" ? "▸" : STEP_MARK[step.kind]}
              </span>
              <span className="tabular-nums text-muted-foreground">
                {new Date(step.at).toLocaleTimeString(undefined, { hour12: false })}
              </span>
              <span className={cn("truncate", STEP_TONE[step.kind])}>{stepLine(step)}</span>
              <span className="truncate text-muted-foreground">{step.model}</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
