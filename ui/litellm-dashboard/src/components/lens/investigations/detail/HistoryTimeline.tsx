"use client";

import { useState } from "react";

import { DotFieldCanvas, DotFieldRoot } from "@/components/shared/dotField/DotField";
import type { DotColumn } from "@/components/shared/dotField/dots";

import { when } from "../../model/format";
import type { Job } from "../../model/types";

const UNSUCCESSFUL_OPACITY = 0.35;

export interface CheckColumn extends DotColumn {
  job: Job | null;
}

const affectedRuns = (job: Job): number => job.assessments.filter((a) => a.issue_checks.length > 0).length;

/** One column per check, oldest on the left, padded with empty slots so spacing stays stable as checks accrue. */
export function checkColumns(jobs: readonly Job[], slots: number): readonly CheckColumn[] {
  const recent = jobs.slice(0, slots).reverse();
  const padding = Array.from(
    { length: slots - recent.length },
    (): CheckColumn => ({
      job: null,
      total: 0,
      failed: 0,
      series: [],
    }),
  );
  const checks = recent.map(
    (job): CheckColumn => ({
      job,
      total: job.coverage.screened,
      failed: affectedRuns(job),
      series: [],
      opacity: job.status === "failed" || job.status === "cancelled" ? UNSUCCESSFUL_OPACITY : 1,
    }),
  );
  return [...padding, ...checks];
}

const pct = (value: number): string => `${value * 100}%`;

export interface HistoryTimelineProps {
  readonly jobs: readonly Job[];
  readonly slots: number;
  readonly onOpen: (jobId: string) => void;
}

/** Every check as a dot column: reviewed traces in grey, affected traces in red, failed checks faded. */
export function HistoryTimeline({ jobs, slots, onOpen }: HistoryTimelineProps) {
  const columns = checkColumns(jobs, slots);
  const [hover, setHover] = useState<number | null>(null);
  const hovered = hover === null ? null : columns[hover];
  return (
    <div className="relative select-none" data-testid="history-timeline">
      <DotFieldRoot columns={columns} hover={hover} className="flex" onPointerLeave={() => setHover(null)}>
        <DotFieldCanvas />
        {columns.map(({ job }, i) =>
          job ? (
            <button
              key={job.id}
              type="button"
              aria-label={`Check at ${when(job.created_at)}`}
              className="relative h-full flex-1 focus-visible:outline-2 focus-visible:outline-ring"
              onPointerEnter={() => setHover(i)}
              onFocus={() => setHover(i)}
              onBlur={() => setHover(null)}
              onClick={() => onOpen(job.id)}
            />
          ) : (
            <span key={`empty-${i}`} className="h-full flex-1" />
          ),
        )}
      </DotFieldRoot>
      {hovered?.job && hover !== null && (
        <div
          role="tooltip"
          className="pointer-events-none absolute top-full z-floating mt-1 rounded-md border border-border bg-popover px-2.5 py-1.5 font-mono text-xs text-popover-foreground shadow-md"
          style={{ left: pct(Math.min(0.75, hover / slots)) }}
        >
          <div>{when(hovered.job.created_at)}</div>
          <div>
            {hovered.total} reviewed, {hovered.failed} affected
          </div>
          {hovered.job.status !== "completed" && <div className="capitalize">{hovered.job.status}</div>}
        </div>
      )}
    </div>
  );
}
