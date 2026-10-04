"use client";

import { StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";

import type { Conclusion, Phase, Playback } from "../../model/live";
import type { Job, Review } from "../../model/types";
import { ConclusionsPanel } from "./ConclusionsPanel";
import { ReadingPanel } from "./ReadingPanel";
import { ReviewQueue } from "./ReviewQueue";
import { ModelName } from "./LiveStrip";

const STATUS: Record<Job["status"], { label: string; tone: StatusTone }> = {
  queued: { label: "Queued", tone: "neutral" },
  running: { label: "Running", tone: "neutral" },
  completed: { label: "Done", tone: "neutral" },
  failed: { label: "Failed", tone: "error" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};
const SECTION_HEAD =
  "flex h-8 items-center justify-between border-y border-border bg-muted/40 px-4 text-[10px] tracking-[0.08em] text-muted-foreground uppercase";

export function LiveDrawer({
  open,
  onClose,
  name,
  model,
  job,
  playback,
  live,
  focused,
  following,
  phase,
  groups,
  scope,
  onPick,
  onFollow,
}: {
  open: boolean;
  onClose: () => void;
  name: string;
  model: string;
  job: Job;
  playback: Pick<Playback, "played" | "current">;
  live: boolean;
  focused: Review | null;
  following: boolean;
  phase: Phase;
  groups: readonly Conclusion[];
  scope: string;
  onPick: (review: Review) => void;
  onFollow: () => void;
}) {
  const status = STATUS[job.status];
  const shownPhase = following ? phase : { span: -1, typed: Number.MAX_SAFE_INTEGER, verdict: true };
  return (
    <Sheet open={open} onOpenChange={(value) => !value && onClose()}>
      <SheetContent className="w-full gap-0 overflow-y-auto data-[side=right]:sm:max-w-[600px]">
        <SheetHeader className="gap-1.5 border-b">
          <SheetTitle className="pr-8 text-base">{name}</SheetTitle>
          <SheetDescription render={<div />} className="flex flex-wrap items-center gap-2">
            <StatusBadge tone={status.tone} label={status.label} />
            <ModelName model={model} />
            {!following && live && (
              <Button variant="outline" size="xs" className="ml-auto rounded-full" onClick={onFollow}>
                Back to live
              </Button>
            )}
          </SheetDescription>
        </SheetHeader>
        <div className="py-3">
          {focused ? (
            <ReadingPanel key={following ? "live" : focused.execution_id} review={focused} phase={shownPhase} />
          ) : (
            <p className="px-4 py-6 text-[12px] text-muted-foreground">Waiting for the first trace review.</p>
          )}
        </div>
        <div className={SECTION_HEAD}>
          <span>Conclusions</span>
        </div>
        <ConclusionsPanel groups={groups} scope={scope} />
        <div className={SECTION_HEAD}>
          <span>Traces</span>
          <span className="font-mono tracking-normal normal-case">{job.reviewed}</span>
        </div>
        <ReviewQueue playback={playback} live={live} focused={focused} onPick={onPick} />
      </SheetContent>
    </Sheet>
  );
}
