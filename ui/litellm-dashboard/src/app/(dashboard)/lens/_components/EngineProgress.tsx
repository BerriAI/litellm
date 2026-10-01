"use client";

import { useEffect, useState } from "react";
import { Check, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { analysisElapsed, analysisProgress, nextCheckStatus, type Engine, type Job } from "./engineData";

const steps = ["Review runs", "Find patterns", "Check evidence"];

export function EngineProgress({ job, onCancel }: { job: Job; onCancel?: () => void }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const progress = analysisProgress(job);
  const percent = progress.total ? Math.min(100, (progress.done / progress.total) * 100) : undefined;

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
      <ol aria-label="Analysis stages" className="grid grid-cols-3 gap-2">
        {steps.map((label, index) => (
          <li key={label} aria-current={index === progress.step ? "step" : undefined} className="space-y-2">
            <div className={`h-1 rounded-full ${index <= progress.step ? "bg-foreground" : "bg-border"}`} />
            <span
              className={`flex items-center gap-1 text-xs ${index === progress.step ? "font-medium" : "text-muted-foreground"}`}
            >
              {index < progress.step && <Check aria-label="Complete" className="size-3 shrink-0" />}
              {label}
            </span>
          </li>
        ))}
      </ol>
      <div className="space-y-2">
        <p className="text-xs text-muted-foreground">{progress.detail}</p>
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
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
        <span>You can leave this page. Analysis continues in the background.</span>
        {onCancel && (
          <Button variant="ghost" size="sm" onClick={onCancel}>
            Cancel analysis
          </Button>
        )}
      </div>
    </section>
  );
}

export function NextCheck({ engine }: { engine: Engine }) {
  const [now, setNow] = useState(Date.now);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 15000);
    return () => window.clearInterval(timer);
  }, []);
  const label = nextCheckStatus(engine, now);
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
