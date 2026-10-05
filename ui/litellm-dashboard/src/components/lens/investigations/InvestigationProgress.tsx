"use client";

import { useNow } from "@/hooks/useNow";

import { Button } from "@/components/ui/button";
import {
  analysisElapsed,
  analysisFraction,
  analysisPace,
  analysisProgress,
  analysisStages,
  remainingLabel,
  stageDurations,
} from "../model/progress";
import { useProgressSamples } from "./useProgressSamples";
import { durationText } from "../model/format";
import { type Job } from "../model/types";
import { cn } from "@/lib/cva.config";
import { QueueReasonText } from "./QueueReasonText";
import { useQueueReason, type QueueContext } from "./useQueueReason";

const steps = ["review runs", "find patterns", "check evidence"];
const markers = { done: "✓", active: "▸", todo: "·" };
const blocks = 32;
const QUEUED_TITLE = { busy: "Waiting in line", no_worker: "Waiting for a worker", starting: "Picking up" } as const;

function stageState(index: number, current: number): keyof typeof markers {
  if (index < current) return "done";
  return index === current ? "active" : "todo";
}

export function InvestigationProgress({
  job,
  queue,
  onCancel,
}: {
  job: Job;
  queue?: QueueContext;
  onCancel?: () => void;
}) {
  const now = useNow(1000);
  const reason = useQueueReason(job, queue);
  const progress = analysisProgress(job);
  const fraction = analysisFraction(progress);
  const samples = useProgressSamples(progress);
  const pace = analysisPace(samples, now);
  const percent = Math.round(fraction * 100);
  const queued = progress.step < 0;
  const counts = analysisStages(job);
  const durations = stageDurations(samples, job.created_at, now);
  const filled = Math.round(fraction * blocks);
  const stats = [
    ["eta", remainingLabel(pace.secondsLeft)],
    ["rate", pace.perMinute === null ? "–" : `${Math.round(pace.perMinute)}/min`],
    ["elapsed", analysisElapsed(job.created_at, now)],
  ];

  return (
    <section aria-label="Analysis progress" className="space-y-3 rounded-md border bg-muted/40 px-4 py-3 text-sm">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate" role="status">
          <span className="font-medium">{reason ? QUEUED_TITLE[reason.kind] : progress.title}</span>
          {!queued && <span className="text-muted-foreground"> · {progress.detail}</span>}
        </span>
        {onCancel && (
          <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={onCancel}>
            Cancel
          </Button>
        )}
      </div>
      <div className="max-w-2xl space-y-1.5 pl-3">
        <div className="flex items-center gap-2">
          <span aria-hidden="true" className="text-muted-foreground">
            [
          </span>
          <div
            role="progressbar"
            aria-label="Investigation progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent}
            aria-valuetext={queued ? progress.title : `${progress.title}: ${progress.detail}`}
            className="flex h-3.5 min-w-0 flex-1 gap-px"
          >
            {Array.from({ length: blocks }, (_, index) => (
              <span
                key={index}
                data-state={index < filled ? "active" : "inactive"}
                className={cn(
                  "flex-1",
                  index < filled ? "bg-foreground" : "bg-foreground/10",
                  queued && index < blocks / 3 && "motion-safe:animate-pulse bg-foreground/30",
                )}
              />
            ))}
          </div>
          <span aria-hidden="true" className="text-muted-foreground">
            ]
          </span>
          <span className="w-10 text-right tabular-nums">{percent}%</span>
        </div>
        <ol aria-label="Analysis stages" className="space-y-0.5">
          {steps.map((label, index) => {
            const state = stageState(index, progress.step);
            const { done, total } = counts[index];
            const seconds = durations[index];
            return (
              <li
                key={label}
                data-state={state}
                aria-current={state === "active" ? "step" : undefined}
                className="grid grid-cols-[1rem_minmax(0,9rem)_6rem_auto] items-baseline gap-2 tabular-nums data-[state=done]:text-foreground data-[state=active]:font-medium data-[state=active]:text-foreground data-[state=todo]:text-muted-foreground"
              >
                <span aria-hidden="true">{markers[state]}</span>
                <span className="truncate">{label}</span>
                <span className="text-right text-muted-foreground">
                  {state === "todo" || !total ? "–" : `${Math.min(done, total)}/${total}`}
                </span>
                <span className="text-muted-foreground">{seconds === null ? "" : durationText(seconds)}</span>
              </li>
            );
          })}
        </ol>
        <div className="flex flex-wrap gap-x-5 gap-y-1 pt-1 tabular-nums">
          {queued ? (
            <span className="text-muted-foreground">
              {reason ? <QueueReasonText reason={reason} onConnect={queue?.onConnect} /> : progress.detail}
            </span>
          ) : (
            stats.map(([key, value]) => (
              <span key={key}>
                <span className="text-muted-foreground">{key}</span> {value}
              </span>
            ))
          )}
        </div>
      </div>
    </section>
  );
}
