"use client";

import { useState, type ReactNode } from "react";

import { StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";

import type { Conclusion, InFlight } from "../../model/live";
import type { Job, Review } from "../../model/types";
import { ConclusionsPanel } from "./ConclusionsPanel";
import { ModelName } from "./LiveStrip";
import { TraceList } from "./TraceList";

const STATUS: Record<Job["status"], { label: string; tone: StatusTone }> = {
  queued: { label: "Queued", tone: "neutral" },
  running: { label: "Running", tone: "neutral" },
  completed: { label: "Done", tone: "neutral" },
  failed: { label: "Failed", tone: "error" },
  cancelled: { label: "Cancelled", tone: "neutral" },
};
const PANE_TITLE = "text-[12px] font-semibold text-foreground";

export function LiveDrawer({
  open,
  onClose,
  name,
  model,
  status,
  reviewed,
  reviews,
  now,
  reading,
  groups,
  scope,
  waiting,
}: {
  open: boolean;
  onClose: () => void;
  name: string;
  model: string;
  status: Job["status"];
  reviewed: number;
  reviews: readonly Review[];
  now: string | null;
  reading: readonly InFlight[];
  groups: readonly Conclusion[];
  scope: string;
  waiting?: ReactNode;
}) {
  const [group, setGroup] = useState<string | null>(null);
  const badge = STATUS[status];
  return (
    <Sheet open={open} onOpenChange={(value) => !value && onClose()}>
      <SheetContent className="flex h-full w-full flex-col gap-0 data-[side=right]:sm:max-w-[min(1200px,94vw)]">
        <SheetHeader className="shrink-0 gap-1.5 border-b px-5">
          <SheetTitle className="pr-10 text-base">{name}</SheetTitle>
          <SheetDescription render={<div />} className="flex flex-wrap items-center gap-2.5 pr-10">
            <StatusBadge tone={badge.tone} label={badge.label} />
            <ModelName model={model} size="md" />
          </SheetDescription>
        </SheetHeader>
        <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[minmax(0,55fr)_minmax(0,45fr)]">
          <section aria-label="Traces" className="min-h-0 overflow-y-auto border-b px-4 py-4 md:border-r md:border-b-0">
            <div className="mb-2 flex items-baseline justify-between gap-3 px-2">
              <h2 className={PANE_TITLE}>Traces</h2>
              <span className="text-[11px] tabular-nums text-muted-foreground">
                {now ?? `${reviewed} reviewed`}
                {reviewed > reviews.length && reviews.length ? ` · showing latest ${reviews.length}` : ""}
              </span>
            </div>
            {reviews.length || reading.length ? (
              <TraceList reading={reading} reviews={reviews} model={model} group={group} />
            ) : (
              <div className="flex flex-col gap-1 px-2 py-6 text-[12px] text-muted-foreground">
                <p className="text-[13px] text-foreground">Waiting for the first trace…</p>
                {waiting}
              </div>
            )}
          </section>
          <section aria-label="Conclusions so far" className="flex min-h-0 flex-col gap-3 overflow-y-auto px-5 py-4">
            <h2 className={PANE_TITLE}>Conclusions so far</h2>
            <ConclusionsPanel
              groups={groups}
              total={reviews.length}
              scope={scope}
              selected={group}
              onSelect={setGroup}
            />
          </section>
        </div>
      </SheetContent>
    </Sheet>
  );
}
