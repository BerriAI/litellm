"use client";

import { Check } from "lucide-react";

import { Button } from "@/components/ui/button";
import { analysisFraction, analysisPace, analysisProgress, remainingLabel } from "../model/progress";
import { useProgressSamples } from "./useProgressSamples";
import { type Job } from "../model/types";

const STAGES = ["Review runs", "Find patterns", "Check evidence"];

function stageState(index: number, current: number) {
  if (index < current) return "done";
  return index === current ? "active" : "todo";
}

export function InvestigationProgress({ job, now, onCancel }: { job: Job; now: number; onCancel?: () => void }) {
  const progress = analysisProgress(job);
  const percent = Math.round(analysisFraction(progress) * 100);
  const samples = useProgressSamples(progress);
  const { secondsLeft } = analysisPace(samples, now);
  const queued = progress.step < 0;
  return (
    <section aria-label="Analysis progress" className="space-y-3">
      <div className="flex items-baseline justify-between gap-3">
        <p role="status" className="min-w-0 text-sm">
          <span className="font-medium">{progress.title}</span>
          <span className="text-muted-foreground"> · {progress.detail}</span>
        </p>
        <span className="shrink-0 text-sm tabular-nums text-muted-foreground">
          {queued ? "" : `${percent}% · ${remainingLabel(secondsLeft)} left`}
        </span>
      </div>
      <div
        role="progressbar"
        aria-label="Investigation progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={queued ? progress.title : `${progress.title}: ${progress.detail}`}
        className="h-1.5 overflow-hidden rounded-full bg-muted"
      >
        <div
          data-state={queued ? "queued" : "running"}
          className="h-full rounded-full bg-primary transition-[width] duration-500 data-[state=queued]:w-1/4 data-[state=queued]:animate-pulse data-[state=queued]:bg-muted-foreground/40"
          style={queued ? undefined : { width: `${percent}%` }}
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <ol aria-label="Analysis stages" className="flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {STAGES.map((label, index) => {
            const state = stageState(index, progress.step);
            return (
              <li
                key={label}
                data-state={state}
                aria-current={state === "active" ? "step" : undefined}
                className="flex items-center gap-1.5 text-muted-foreground data-[state=active]:font-medium data-[state=active]:text-foreground"
              >
                <span
                  data-state={state}
                  className="flex size-5 items-center justify-center rounded-full border tabular-nums data-[state=active]:border-primary data-[state=active]:text-primary data-[state=done]:border-transparent data-[state=done]:bg-primary data-[state=done]:text-primary-foreground"
                >
                  {state === "done" ? <Check className="size-3" /> : index + 1}
                </span>
                {label}
              </li>
            );
          })}
        </ol>
        {onCancel && (
          <Button variant="ghost" size="xs" onClick={onCancel}>
            Cancel
          </Button>
        )}
      </div>
    </section>
  );
}
