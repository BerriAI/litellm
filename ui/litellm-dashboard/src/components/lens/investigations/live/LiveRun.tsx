"use client";

import { useMemo, useRef, useState, type CSSProperties } from "react";

import { Button } from "@/components/ui/button";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";

import { sampleBatch } from "../../__fixtures__/reviews";
import { durationText, money } from "../../model/format";
import { modelsUsed } from "../../model/inbox";
import {
  analysisModel,
  conclusions,
  isCompact,
  liveStats,
  outcome,
  providerAccent,
  providerOf,
  rateLabel,
  reviewKey,
  shownCount,
  tokenLabel,
  verdictLine,
} from "../../model/live";
import { analysisProgress } from "../../model/progress";
import type { Job, Review } from "../../model/types";
import { ConclusionsPanel, type Arrival } from "./ConclusionsPanel";
import { ProviderBadge } from "./ProviderBadge";
import { ReadingPanel } from "./ReadingPanel";
import { ReviewQueue } from "./ReviewQueue";
import { useReviewPlayback } from "./useReviewPlayback";

const NEUTRAL_ACCENT = "var(--foreground)";
const SAMPLE_POLL_MS = 2000;
const SAMPLE_TOTAL = 60;
const COLUMN_HEAD =
  "flex h-9 shrink-0 items-center justify-between gap-2 border-b px-3.5 text-[10px] tracking-[0.08em] text-muted-foreground uppercase";

function useArrival(review: Review | null, verdict: boolean, compact: boolean): Arrival | null {
  const issue = review && verdict && !compact ? review.verdicts.find((v) => v.kind === "issue") : undefined;
  const key = review && issue ? reviewKey(review) : "";
  const checkId = issue?.check_id ?? "";
  const text = review ? verdictLine(review) : "";
  return useMemo(() => (key ? { key, checkId, text } : null), [key, checkId, text]);
}

function Status({ job }: { job: Job }) {
  if (job.status === "queued") return <span className="text-xs text-muted-foreground">Queued</span>;
  if (job.status === "running")
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground">
        <span aria-hidden="true">●</span> Running
      </span>
    );
  if (job.status === "completed")
    return (
      <span className="inline-flex items-center gap-1.5 rounded-md bg-muted px-2.5 py-1 text-xs font-medium">
        <span aria-hidden="true">✓</span> Done
      </span>
    );
  return <span className="text-xs text-muted-foreground capitalize">{job.status}</span>;
}

function useSampleReviews(enabled: boolean): readonly Review[] {
  const now = useNow(SAMPLE_POLL_MS);
  const [startedAt] = useState(now);
  return useMemo(() => {
    if (!enabled) return [];
    const polls = Math.floor((now - startedAt) / SAMPLE_POLL_MS);
    return Array.from({ length: polls + 1 }, (_, poll) => sampleBatch(poll, poll < 3 ? 1 : 3))
      .flat()
      .slice(0, SAMPLE_TOTAL);
  }, [enabled, now, startedAt]);
}

export function LiveRun({ job, onCancel }: { job: Job; onCancel?: () => void }) {
  const live = job.status === "queued" || job.status === "running";
  const sample = live && job.reviews.length === 0;
  const sampled = useSampleReviews(sample);
  const reviews = sample ? sampled : job.reviews;
  const playback = useReviewPlayback(reviews, live);
  const { current, played, pending, phase, duration } = playback;
  const now = useNow(1000);
  const verdictRef = useRef<HTMLDivElement>(null);
  const model = analysisModel([current?.model ?? "", ...reviews.map((r) => r.model), ...modelsUsed(job.steps)]);
  const accent = providerAccent(providerOf(model)) ?? NEUTRAL_ACCENT;
  const decided = phase.verdict && current ? [...played, current] : played;
  const groups = conclusions(decided, job.settings.checks);
  const arrival = useArrival(current, phase.verdict, isCompact(duration));
  const stats = liveStats(job, now);
  const reviewed = sample ? decided.length : shownCount(job.reviewed, playback);
  const selected = sample ? SAMPLE_TOTAL : job.coverage.selected;
  const percent = live ? Math.min(100, Math.round((reviewed / Math.max(1, selected)) * 100)) : 100;
  const issues = decided.filter((r) => outcome(r) === "issue").length;
  const progress = analysisProgress(job);

  return (
    <section
      aria-label="Live review"
      style={{ "--provider": accent } as CSSProperties}
      className="flex flex-col overflow-hidden rounded-xl border bg-background"
    >
      <header className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 py-2.5">
        <div className="flex min-w-0 items-center gap-3">
          <Status job={job} />
          <span className="truncate text-xs text-muted-foreground">
            {live ? progress.title : `Reviewed ${reviewed} ${reviewed === 1 ? "trace" : "traces"}`}
            {sample && " · preview data until the worker reports reviews"}
          </span>
        </div>
        <div className="flex min-w-0 items-center gap-2">
          <ProviderBadge model={model} />
          {live && onCancel && (
            <Button variant="ghost" size="xs" onClick={onCancel}>
              Cancel
            </Button>
          )}
        </div>
      </header>
      <div
        role="progressbar"
        aria-label="Traces reviewed"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={`${reviewed} of ${selected} traces reviewed`}
        className="h-[3px] bg-muted"
      >
        <div
          className="h-full bg-(--provider) transition-[width] duration-300 motion-reduce:transition-none"
          style={{ width: `${percent}%` }}
        />
      </div>
      <div className="grid min-h-0 grid-cols-1 md:h-[440px] md:grid-cols-[minmax(0,15rem)_minmax(0,1fr)] xl:grid-cols-[17rem_minmax(0,1fr)_20rem]">
        <div className="flex min-h-0 flex-col border-b md:border-r md:border-b-0">
          <div className={COLUMN_HEAD}>
            <span>Traces</span>
            <span className="font-mono tracking-normal normal-case">
              {reviewed} / {selected}
            </span>
          </div>
          <ReviewQueue played={played} current={current} pending={pending} live={live} />
        </div>
        <div className="flex min-h-0 flex-col border-b xl:border-r xl:border-b-0">
          <div className={COLUMN_HEAD}>
            <span>{live ? "Reading trace" : "Last trace"}</span>
            <span className="font-mono tracking-normal normal-case">{rateLabel(stats.perSecond)}</span>
          </div>
          <ReadingPanel review={current} phase={phase} tokens={tokenLabel(stats.tokens)} verdictRef={verdictRef} />
        </div>
        <div className="flex min-h-0 flex-col bg-muted/30 md:col-span-2 xl:col-span-1">
          <div className={COLUMN_HEAD}>
            <span>Conclusions</span>
            <span className={cn("font-mono tracking-normal normal-case", issues > 0 && "text-[#e5484d]")}>
              {issues} {issues === 1 ? "problem" : "problems"}
            </span>
          </div>
          <ConclusionsPanel groups={groups} reviewed={decided.length} arrival={arrival} verdictRef={verdictRef} />
          <footer
            aria-label="Run stats"
            className="flex shrink-0 flex-wrap justify-between gap-x-4 gap-y-1 border-t bg-background px-3.5 py-2 font-mono text-[11px] text-muted-foreground"
          >
            <span>{rateLabel(stats.perSecond)}</span>
            <span>{tokenLabel(stats.tokens)}</span>
            <span>{money(stats.cost)}</span>
            <span>{durationText(stats.elapsedSeconds)}</span>
          </footer>
        </div>
      </div>
    </section>
  );
}
