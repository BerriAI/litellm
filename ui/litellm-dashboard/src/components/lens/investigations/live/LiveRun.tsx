"use client";

import { ChevronDown, Loader2 } from "lucide-react";
import { useMemo, useRef } from "react";

import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { Button } from "@/components/ui/button";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";

import { durationText, money } from "../../model/format";
import { modelsUsed } from "../../model/inbox";
import {
  analysisModel,
  conclusions,
  isCompact,
  liveStats,
  outcome,
  providerOf,
  reviewKey,
  shownCount,
  tickerLine,
  verdictLine,
} from "../../model/live";
import type { Job, Review } from "../../model/types";
import { ConclusionsPanel, type Arrival } from "./ConclusionsPanel";
import { ReadingPanel } from "./ReadingPanel";
import { ReviewQueue } from "./ReviewQueue";
import { useReviewPlayback } from "./useReviewPlayback";
import { useShowWork } from "./useShowWork";

const COLUMN_HEAD =
  "sticky top-0 z-sticky flex h-8 shrink-0 items-center justify-between gap-2 border-b border-border bg-muted/40 px-3 text-[10px] tracking-[0.08em] text-muted-foreground uppercase backdrop-blur";
const COLUMN = "flex max-h-[320px] min-h-0 flex-col overflow-y-auto";

function useArrival(review: Review | null, verdict: boolean, enabled: boolean): Arrival | null {
  const issue = review && verdict && enabled ? review.verdicts.find((v) => v.kind === "issue") : undefined;
  const key = review && issue ? reviewKey(review) : "";
  const checkId = issue?.check_id ?? "";
  const text = review ? verdictLine(review) : "";
  return useMemo(() => (key ? { key, checkId, text } : null), [key, checkId, text]);
}

function Model({ model }: { model: string }) {
  if (!model) return null;
  const provider = providerOf(model);
  return (
    <span data-testid="live-run-model" className="inline-flex shrink-0 items-center gap-1.5 text-foreground">
      {provider && <ProviderLogo provider={provider} className="size-4 shrink-0" />}
      <span className="font-mono text-[11px]">{model}</span>
    </span>
  );
}

function Ticker({ review }: { review: Review | null }) {
  return (
    <span className="relative h-4 min-w-0 flex-1 overflow-hidden text-muted-foreground">
      {review && (
        <span
          key={reviewKey(review)}
          data-testid="live-run-ticker"
          className="absolute inset-0 truncate motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300"
        >
          {tickerLine(review)}
        </span>
      )}
    </span>
  );
}

export function LiveRun({ job, onCancel }: { job: Job; onCancel?: () => void }) {
  const live = job.status === "queued" || job.status === "running";
  const [open, toggle] = useShowWork();
  const playback = useReviewPlayback(job.reviews, live);
  const { current, played, phase, duration, replay } = playback;
  const now = useNow(1000);
  const verdictRef = useRef<HTMLDivElement>(null);
  const model = analysisModel([...job.reviews.map((r) => r.model), ...modelsUsed(job.steps), job.settings.model]);
  const decided = phase.verdict && current ? [...played, current] : played;
  const groups = conclusions(decided, job.settings.checks);
  const arrival = useArrival(current, phase.verdict, open && !isCompact(duration));
  const stats = liveStats(job, now);
  const reviewed = live ? shownCount(job.reviewed, playback) : job.reviewed;
  const selected = job.coverage.selected;
  const percent = live ? Math.min(100, Math.round((reviewed / Math.max(1, selected)) * 100)) : 100;
  const issues = (live ? decided : job.reviews).filter((r) => outcome(r) === "issue").length;
  const showWork = () => {
    if (!open && !live) replay();
    toggle();
  };
  const issueText = `${issues} ${issues === 1 ? "issue" : "issues"}`;

  return (
    <section aria-label="Analysis progress" className="overflow-hidden rounded-md border bg-card">
      <div className="flex h-9 items-center gap-3 px-3 text-[12px]">
        {live ? (
          <>
            <Loader2 aria-hidden="true" className="size-3.5 shrink-0 text-muted-foreground motion-safe:animate-spin" />
            <Model model={model} />
            <Ticker review={current} />
            <span role="status" className="shrink-0 text-muted-foreground tabular-nums">
              {reviewed} of {selected} traces · {issueText} · {money(stats.cost)} · {durationText(stats.elapsedSeconds)}
            </span>
          </>
        ) : (
          <span role="status" className="flex min-w-0 flex-1 items-center gap-1.5 text-muted-foreground">
            <span className="shrink-0">Reviewed {reviewed} traces with</span>
            <Model model={model} />
            <span className="truncate tabular-nums">
              in {durationText(stats.elapsedSeconds)} · {issueText} · {money(stats.cost)}
            </span>
          </span>
        )}
        {live && onCancel && (
          <Button variant="ghost" size="xs" className="shrink-0" onClick={onCancel}>
            Cancel
          </Button>
        )}
        <Button
          variant="ghost"
          size="xs"
          aria-expanded={open}
          className="shrink-0 text-muted-foreground"
          onClick={showWork}
        >
          {open ? "Hide work" : "Show work"}
          <ChevronDown className={cn("transition-transform duration-150", open && "rotate-180")} />
        </Button>
      </div>
      <div
        role="progressbar"
        aria-label="Traces reviewed"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
        aria-valuetext={`${reviewed} of ${selected} traces reviewed`}
        className="h-px bg-foreground/10"
      >
        <div
          className="h-full bg-foreground/60 transition-[width] duration-300 motion-reduce:transition-none"
          style={{ width: `${percent}%` }}
        />
      </div>
      {open && (
        <div className="grid grid-cols-1 md:grid-cols-[minmax(0,15rem)_minmax(0,1fr)] xl:grid-cols-[16rem_minmax(0,1fr)_20rem]">
          <div className={`${COLUMN} border-b md:border-r md:border-b-0`}>
            <div className={COLUMN_HEAD}>
              <span>Traces</span>
              <span className="font-mono tracking-normal normal-case">{reviewed}</span>
            </div>
            <ReviewQueue playback={playback} live={live} />
          </div>
          <div className={`${COLUMN} border-b xl:border-r xl:border-b-0`}>
            <div className={COLUMN_HEAD}>
              <span>{live ? "Reading trace" : "Last trace"}</span>
            </div>
            <ReadingPanel review={current} phase={phase} verdictRef={verdictRef} />
          </div>
          <div className={`${COLUMN} md:col-span-2 xl:col-span-1`}>
            <div className={COLUMN_HEAD}>
              <span>Conclusions</span>
              <span className="font-mono tracking-normal normal-case">{issueText}</span>
            </div>
            <ConclusionsPanel groups={groups} reviewed={decided.length} arrival={arrival} verdictRef={verdictRef} />
          </div>
        </div>
      )}
    </section>
  );
}
