"use client";

import { useMemo, useRef } from "react";

import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { Button } from "@/components/ui/button";
import { useNow } from "@/hooks/useNow";

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
  tokenLabel,
  verdictLine,
} from "../../model/live";
import type { Job, Review } from "../../model/types";
import { ConclusionsPanel, type Arrival } from "./ConclusionsPanel";
import { ReadingPanel } from "./ReadingPanel";
import { ReviewQueue } from "./ReviewQueue";
import { useReviewPlayback } from "./useReviewPlayback";

const STATUS: Record<Job["status"], { label: string; tone: StatusTone }> = {
  queued: { label: "Queued", tone: "neutral" },
  running: { label: "Running", tone: "neutral" },
  completed: { label: "Done", tone: "neutral" },
  failed: { label: "Failed", tone: "error" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};
const COLUMN_HEAD =
  "sticky top-0 z-sticky flex h-8 shrink-0 items-center justify-between gap-2 border-b border-border bg-muted/40 px-3 text-[10px] tracking-[0.08em] text-muted-foreground uppercase backdrop-blur";
const COLUMN = "flex min-h-0 flex-col overflow-y-auto";

function useArrival(review: Review | null, verdict: boolean, compact: boolean): Arrival | null {
  const issue = review && verdict && !compact ? review.verdicts.find((v) => v.kind === "issue") : undefined;
  const key = review && issue ? reviewKey(review) : "";
  const checkId = issue?.check_id ?? "";
  const text = review ? verdictLine(review) : "";
  return useMemo(() => (key ? { key, checkId, text } : null), [key, checkId, text]);
}

function ModelLabel({ model }: { model: string }) {
  if (!model) return null;
  const provider = providerOf(model);
  return (
    <span data-testid="live-run-model" className="inline-flex min-w-0 items-center gap-1.5 text-muted-foreground">
      {provider && <ProviderLogo provider={provider} className="size-3.5 shrink-0" />}
      <span className="truncate font-mono text-[11px]">{model}</span>
    </span>
  );
}

export function LiveRun({ job, onCancel }: { job: Job; onCancel?: () => void }) {
  const live = job.status === "queued" || job.status === "running";
  const playback = useReviewPlayback(job.reviews, live);
  const { current, played, phase, duration } = playback;
  const now = useNow(1000);
  const verdictRef = useRef<HTMLDivElement>(null);
  const model = analysisModel([...job.reviews.map((r) => r.model), ...modelsUsed(job.steps), job.settings.model]);
  const decided = phase.verdict && current ? [...played, current] : played;
  const groups = conclusions(decided, job.settings.checks);
  const arrival = useArrival(current, phase.verdict, isCompact(duration));
  const stats = liveStats(job, now);
  const reviewed = shownCount(job.reviewed, playback);
  const selected = job.coverage.selected;
  const percent = live ? Math.min(100, Math.round((reviewed / Math.max(1, selected)) * 100)) : 100;
  const issues = decided.filter((r) => outcome(r) === "issue").length;
  const status = STATUS[job.status];

  return (
    <section aria-label="Analysis progress" className="overflow-hidden rounded-md border bg-card">
      <header className="flex min-h-10 flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b px-3 py-1.5 text-[12px]">
        <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
          <StatusBadge tone={status.tone} label={status.label} dataTestId="live-run-status" />
          <span role="status" className="truncate text-foreground">
            {live ? job.stage : "Complete"}
          </span>
          <span className="text-muted-foreground tabular-nums">
            {reviewed} of {selected} traces
          </span>
          <ModelLabel model={model} />
        </div>
        <div className="flex items-center gap-4 font-mono text-[11px] text-muted-foreground tabular-nums">
          <span title="Elapsed">{durationText(stats.elapsedSeconds)}</span>
          <span title="Tokens">{tokenLabel(stats.tokens)}</span>
          <span title="Cost">{money(stats.cost)}</span>
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
        className="h-0.5 bg-foreground/10"
      >
        <div
          className="h-full bg-foreground transition-[width] duration-300 motion-reduce:transition-none"
          style={{ width: `${percent}%` }}
        />
      </div>
      <div className="grid grid-cols-1 md:h-[420px] md:grid-cols-[minmax(0,15rem)_minmax(0,1fr)] xl:grid-cols-[16rem_minmax(0,1fr)_20rem]">
        <div className={`${COLUMN} max-h-[420px] border-b md:border-r md:border-b-0`}>
          <div className={COLUMN_HEAD}>
            <span>Traces</span>
            <span className="font-mono tracking-normal normal-case">{reviewed}</span>
          </div>
          <ReviewQueue playback={playback} live={live} />
        </div>
        <div className={`${COLUMN} max-h-[420px] border-b xl:border-r xl:border-b-0`}>
          <div className={COLUMN_HEAD}>
            <span>{live ? "Reading trace" : "Last trace"}</span>
          </div>
          <ReadingPanel review={current} phase={phase} verdictRef={verdictRef} />
        </div>
        <div className={`${COLUMN} max-h-[420px] md:col-span-2 xl:col-span-1`}>
          <div className={COLUMN_HEAD}>
            <span>Conclusions</span>
            <span className="font-mono tracking-normal normal-case">
              {issues} {issues === 1 ? "issue" : "issues"}
            </span>
          </div>
          <ConclusionsPanel groups={groups} reviewed={decided.length} arrival={arrival} verdictRef={verdictRef} />
        </div>
      </div>
    </section>
  );
}
