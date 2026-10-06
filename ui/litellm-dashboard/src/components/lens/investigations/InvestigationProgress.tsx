"use client";

import { useNow } from "@/hooks/useNow";
import {
  analysisElapsed,
  analysisFraction,
  analysisPace,
  analysisProgress,
  analysisStages,
  remainingLabel,
} from "../model/progress";
import { useProgressSamples } from "./useProgressSamples";
import { type Job } from "../model/types";
import { QueueReasonText } from "./QueueReasonText";
import { useQueueReason, type QueueContext } from "./useQueueReason";

export function InvestigationProgress({ job, queue }: { job: Job; queue?: QueueContext }) {
  const now = useNow(1000);
  const reason = useQueueReason(job, queue);
  const progress = analysisProgress(job);
  const stages = analysisStages(job);
  const percent = Math.round(analysisFraction(progress) * 100);
  const { secondsLeft } = analysisPace(useProgressSamples(progress), now);
  const queued = progress.step < 0;
  return (
    <section aria-label="Analysis progress" className="space-y-2">
      <div className="flex items-baseline justify-between gap-3 tabular-nums">
        <span
          role="progressbar"
          aria-label="Investigation progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          aria-valuetext={`${percent}% overall. ${progress.title}: ${progress.detail}`}
          title={`Overall progress: ${stages.map((stage) => `${stage.label} ${stage.weight * 100}%`).join(", ")}`}
          className="text-2xl font-semibold"
        >
          {queued ? "–" : `${percent}%`}
          {!queued && <span className="ml-1.5 text-xs font-normal text-muted-foreground">overall</span>}
        </span>
        <span className="text-xs text-muted-foreground">
          {analysisElapsed(job.created_at, now)} elapsed
          {!queued && ` · ${remainingLabel(secondsLeft)} left`}
        </span>
      </div>
      <ol aria-label="Analysis stages" className="flex gap-1">
        {stages.map((stage) => (
          <li
            key={stage.label}
            data-state={stage.state}
            aria-current={stage.state === "active" ? "step" : undefined}
            className="group min-w-0 space-y-1.5"
            style={{ flexGrow: stage.weight, flexBasis: 0 }}
          >
            <div className="h-2 overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full bg-primary transition-[width] duration-500"
                style={{ width: `${stage.fill * 100}%` }}
              />
            </div>
            <div className="flex items-baseline justify-between gap-2 text-xs text-muted-foreground group-data-[state=active]:text-foreground">
              <span className="truncate group-data-[state=active]:font-medium">{stage.label}</span>
              {stage.state !== "todo" && stage.total > 0 && (
                <span className="shrink-0 tabular-nums">
                  {stage.done.toLocaleString()} / {stage.total.toLocaleString()}
                </span>
              )}
            </div>
          </li>
        ))}
      </ol>
      {reason && (
        <p className="text-xs text-muted-foreground">
          <QueueReasonText reason={reason} onConnect={queue?.onConnect} />
        </p>
      )}
    </section>
  );
}
