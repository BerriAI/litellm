"use client";

import { useEffect, useState } from "react";
import { Check, Circle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  analysisElapsed,
  analysisProgress,
  analysisStages,
  formatDuration,
  nextCheckStatus,
  remainingLabel,
  stageRemaining,
  stageTimings,
  type Lens,
  type Job,
} from "./lensData";

export function LensProgress({ job, onCancel }: { job: Job; onCancel?: () => void }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const progress = analysisProgress(job);
  const timings = stageTimings(job, now);
  const percent = progress.total ? Math.min(100, (progress.done / progress.total) * 100) : undefined;
  const remaining = stageRemaining(progress.done, progress.total, timings[progress.step]);

  return (
    <section aria-label="Analysis progress" className="space-y-4 rounded-xl border bg-muted/30 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-sm font-medium" role="status">
          <Loader2 aria-hidden="true" className="size-4 motion-safe:animate-spin text-muted-foreground" />
          {progress.title}
        </div>
        <span className="text-xs tabular-nums text-muted-foreground">
          {analysisElapsed(job.created_at, now)} elapsed
        </span>
      </div>
      <ol aria-label="Analysis stages" className="space-y-3">
        {analysisStages.map(({ label }, index) => {
          const current = index === progress.step;
          const seconds = timings[index];
          return (
            <li key={label} aria-current={current ? "step" : undefined} className="flex gap-3">
              <StageIcon index={index} step={progress.step} />
              <div className="min-w-0 flex-1 space-y-2">
                <div className="flex items-baseline justify-between gap-2 text-sm">
                  <span
                    className={`${current ? "font-medium" : ""} ${index > progress.step ? "text-muted-foreground" : ""}`}
                  >
                    {label}
                  </span>
                  <span className="text-xs tabular-nums text-muted-foreground">
                    {seconds !== undefined && index <= progress.step ? formatDuration(seconds) : ""}
                  </span>
                </div>
                {current && (
                  <div className="space-y-1.5">
                    <div
                      role="progressbar"
                      aria-label={progress.title}
                      aria-valuemin={0}
                      aria-valuemax={progress.total || undefined}
                      aria-valuenow={progress.total ? Math.min(progress.done, progress.total) : undefined}
                      aria-valuetext={progress.detail}
                      className="h-1.5 overflow-hidden rounded-full bg-muted"
                    >
                      <div
                        className={`h-full rounded-full bg-foreground/70 transition-[width] duration-500 ${percent === undefined ? "motion-safe:animate-pulse" : ""}`}
                        style={{ width: percent === undefined ? "100%" : `${percent}%` }}
                      />
                    </div>
                    <div className="flex flex-wrap justify-between gap-2 text-xs tabular-nums text-muted-foreground">
                      <span>{progress.detail}</span>
                      <span>{timeLeft(remaining, progress.total)}</span>
                    </div>
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        {job.status === "running" && <span>You can leave this page while the investigation runs.</span>}
        {onCancel && (
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel analysis
          </Button>
        )}
      </div>
    </section>
  );
}

function StageIcon({ index, step }: { index: number; step: number }) {
  if (index < step) return <Check aria-label="Complete" className="mt-0.5 size-4 shrink-0 text-foreground" />;
  if (index === step) return <Loader2 aria-hidden="true" className="mt-0.5 size-4 shrink-0 motion-safe:animate-spin" />;
  return <Circle aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-border" />;
}

function timeLeft(remaining: number | undefined, total: number): string {
  if (remaining !== undefined) return remainingLabel(remaining);
  return total ? "Estimating time left" : "";
}

export function NextCheck({ lens }: { lens: Lens }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 15000);
    return () => window.clearInterval(timer);
  }, []);
  const label = nextCheckStatus(lens, now);
  if (!label) return null;
  return <p className="mt-1 text-xs text-muted-foreground">{label}</p>;
}

export function ScanDuration({ job }: { job: Job }) {
  if (!job.finished_at) return null;
  return (
    <span title="Total time, including any wait for an analyzer">
      {" · Took "}
      {analysisElapsed(job.created_at, Date.parse(job.finished_at))}
    </span>
  );
}
