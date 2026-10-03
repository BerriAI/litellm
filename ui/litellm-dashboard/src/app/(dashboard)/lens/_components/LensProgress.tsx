"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  analysisElapsed,
  analysisFraction,
  analysisPace,
  analysisProgress,
  nextCheckStatus,
  remainingLabel,
  stageWeights,
  type Lens,
  type Job,
  type ProgressSample,
} from "./lensData";

const steps = ["Review runs", "Find patterns", "Check evidence"];
const units = ["runs", "batches", "patterns"];

export function LensProgress({ job, onCancel }: { job: Job; onCancel?: () => void }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const progress = analysisProgress(job);
  const fraction = analysisFraction(progress);
  const [samples, setSamples] = useState<ProgressSample[]>([]);
  const latest = samples.at(-1);
  if (!latest || latest.step !== progress.step || latest.done !== progress.done) {
    setSamples([...samples, { at: now, step: progress.step, done: progress.done, fraction }].slice(-120));
  }
  const pace = analysisPace(samples, now);
  const percent = Math.round(fraction * 100);
  const queued = progress.step < 0;
  const status = queued ? progress.title : `${progress.title}: ${progress.detail}`;

  return (
    <section aria-label="Analysis progress" className="space-y-2 rounded-xl border bg-muted/30 p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2 text-sm">
        <span className="min-w-0 font-medium" role="status">
          {status}
        </span>
        <span className="tabular-nums text-muted-foreground">{percent}%</span>
      </div>
      <div
        role="progressbar"
        aria-label="Investigation progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={status}
        className="relative h-2 overflow-hidden rounded-full bg-muted"
      >
        <div
          className={`h-full rounded-full bg-foreground transition-[width] duration-700 ${queued ? "motion-safe:animate-pulse" : ""}`}
          style={{ width: queued ? "33%" : `${Math.max(fraction * 100, 1)}%` }}
        />
        {stageWeights.slice(0, -1).map((_, index) => (
          <span
            key={index}
            aria-hidden="true"
            className="absolute inset-y-0 w-0.5 bg-background"
            style={{ left: `${stageWeights.slice(0, index + 1).reduce((sum, weight) => sum + weight, 0) * 100}%` }}
          />
        ))}
      </div>
      <ol aria-label="Analysis stages" className="flex text-xs">
        {steps.map((label, index) => (
          <li
            key={label}
            aria-current={index === progress.step ? "step" : undefined}
            style={{ width: `${stageWeights[index] * 100}%` }}
            className={`truncate pr-2 ${index === progress.step ? "font-medium text-foreground" : "text-muted-foreground"}`}
          >
            {index < progress.step ? `${label} ✓` : label}
          </li>
        ))}
      </ol>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 pt-1 text-xs text-muted-foreground">
        <span className="tabular-nums">
          {queued
            ? progress.detail
            : [
                remainingLabel(pace.secondsLeft),
                pace.perMinute !== null && `${Math.round(pace.perMinute)} ${units[progress.step]}/min`,
                `${analysisElapsed(job.created_at, now)} elapsed`,
              ]
                .filter(Boolean)
                .join(" · ")}
        </span>
        {onCancel && (
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={onCancel}>
            Cancel
          </Button>
        )}
      </div>
    </section>
  );
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
