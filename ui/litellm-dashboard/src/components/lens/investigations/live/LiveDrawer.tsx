"use client";

import { useState } from "react";

import { StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";

import type { Conclusion, Phase, Playback } from "../../model/live";
import type { Job, Review } from "../../model/types";
import { ConclusionsPanel } from "./ConclusionsPanel";
import { ModelName } from "./LiveStrip";
import { ReadingPanel } from "./ReadingPanel";
import { ReviewQueue } from "./ReviewQueue";

const STATUS: Record<Job["status"], { label: string; tone: StatusTone }> = {
  queued: { label: "Queued", tone: "neutral" },
  running: { label: "Running", tone: "neutral" },
  completed: { label: "Done", tone: "neutral" },
  failed: { label: "Failed", tone: "error" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};
const PANE_TITLE = "text-[12px] font-semibold text-foreground";
const SETTLED: Phase = { span: -1, typed: Number.MAX_SAFE_INTEGER, verdict: true };

export function LiveDrawer({
  open,
  onClose,
  name,
  model,
  status,
  reviewed,
  playback,
  live,
  focused,
  following,
  phase,
  groups,
  decided,
  scope,
  onPick,
  onFollow,
}: {
  open: boolean;
  onClose: () => void;
  name: string;
  model: string;
  status: Job["status"];
  reviewed: number;
  playback: Pick<Playback, "played" | "current">;
  live: boolean;
  focused: Review | null;
  following: boolean;
  phase: Phase;
  groups: readonly Conclusion[];
  decided: number;
  scope: string;
  onPick: (review: Review) => void;
  onFollow: () => void;
}) {
  const [group, setGroup] = useState<string | null>(null);
  const badge = STATUS[status];
  const shown = playback.played.length + (playback.current ? 1 : 0);
  return (
    <Sheet open={open} onOpenChange={(value) => !value && onClose()}>
      <SheetContent className="flex h-full w-full flex-col gap-0 data-[side=right]:sm:max-w-[min(1200px,94vw)]">
        <SheetHeader className="shrink-0 gap-1.5 border-b px-5">
          <SheetTitle className="pr-10 text-base">{name}</SheetTitle>
          <SheetDescription render={<div />} className="flex flex-wrap items-center gap-2.5 pr-10">
            <StatusBadge tone={badge.tone} label={badge.label} />
            <ModelName model={model} size="md" />
            {!following && live && (
              <Button variant="outline" size="xs" className="ml-auto rounded-full" onClick={onFollow}>
                Back to live
              </Button>
            )}
          </SheetDescription>
        </SheetHeader>
        <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[minmax(0,58fr)_minmax(0,42fr)]">
          <section aria-label="Now reviewing" className="min-h-0 overflow-y-auto border-b px-5 py-4 md:border-r md:border-b-0">
            <h2 className={`${PANE_TITLE} mb-3`}>{following && live ? "Now reviewing" : "Trace"}</h2>
            {focused ? (
              <ReadingPanel
                key={following ? "live" : focused.execution_id}
                review={focused}
                phase={following ? phase : SETTLED}
              />
            ) : (
              <p className="py-6 text-[12px] text-muted-foreground">Waiting for the first trace review.</p>
            )}
          </section>
          <section aria-label="Conclusions so far" className="flex min-h-0 flex-col gap-5 overflow-y-auto px-5 py-4">
            <div className="flex flex-col gap-3">
              <h2 className={PANE_TITLE}>Conclusions so far</h2>
              <ConclusionsPanel groups={groups} total={decided} scope={scope} selected={group} onSelect={setGroup} />
            </div>
            <div className="flex flex-col gap-2">
              <div className="flex items-baseline justify-between gap-3">
                <h2 className={PANE_TITLE}>Traces</h2>
                <span className="text-[11px] tabular-nums text-muted-foreground">
                  {reviewed} reviewed{reviewed > shown ? ` · showing latest ${shown}` : ""}
                </span>
              </div>
              <ReviewQueue playback={playback} live={live} focused={focused} group={group} onPick={onPick} />
            </div>
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}
