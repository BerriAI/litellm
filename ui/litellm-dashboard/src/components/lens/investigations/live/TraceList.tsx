"use client";

import { useState } from "react";

import { agoLabel } from "../../model/format";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";

import { briefReasoning, durationLabel, inGroup, newestFirst, outcome, shortVerdict } from "../../model/live";
import type { Review } from "../../model/types";

const LIMIT = 200;
const RED = "text-destructive";
const ROW =
  "grid h-9 w-full grid-cols-[0.75rem_minmax(0,5rem)_minmax(0,1fr)_auto] sm:grid-cols-[0.75rem_minmax(0,8rem)_minmax(0,1fr)_auto] items-center gap-2 px-2 text-left text-xs";

function Expanded({ review }: { review: Review }) {
  const verdicts = review.verdicts.length
    ? review.verdicts
    : [
        {
          check_id: "",
          kind: "pattern" as const,
          summary: review.cannot_assess ? "Not enough evidence to judge" : "No issue observed",
        },
      ];
  return (
    <div className="flex flex-col gap-1.5 px-7 pt-0.5 pb-2.5 text-xs leading-relaxed">
      {review.reasoning && <p className="text-muted-foreground">{briefReasoning(review.reasoning)}</p>}
      <ul className="flex flex-col gap-1">
        {verdicts.map((verdict, index) => (
          <li
            key={`${verdict.check_id}-${index}`}
            className={cn("flex gap-2", verdict.kind === "issue" ? RED : "text-foreground")}
          >
            <span
              aria-hidden="true"
              className={cn(
                "mt-[0.45rem] size-1.5 shrink-0 rounded-full",
                verdict.kind === "issue" ? "bg-destructive" : "bg-muted-foreground/40",
              )}
            />
            <span className="min-w-0">{verdict.summary}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DoneRow({
  review,
  now,
  open,
  onToggle,
}: {
  review: Review;
  now: number;
  open: boolean;
  onToggle: () => void;
}) {
  const result = outcome(review);
  return (
    <>
      <button type="button" aria-expanded={open} onClick={onToggle} className={cn(ROW, "rounded-lg hover:bg-muted/50")}>
        <span
          aria-label={result}
          className={cn("size-1.5 rounded-full", result === "issue" ? "bg-destructive" : "bg-muted-foreground/40")}
        />
        <span className="truncate text-foreground">{review.agent || review.name}</span>
        <span
          className={cn(
            "truncate motion-safe:animate-in motion-safe:fade-in motion-safe:duration-300",
            result === "issue" ? RED : "text-muted-foreground",
          )}
        >
          {shortVerdict(review)}
        </span>
        <span className="text-xs tabular-nums text-muted-foreground">
          {durationLabel(review.duration_ms)} · {agoLabel(Date.parse(review.at), now)}
        </span>
      </button>
      {open && <Expanded review={review} />}
    </>
  );
}

export function TraceList({ reviews, group }: { reviews: readonly Review[]; group: string | null }) {
  const now = useNow(5000);
  const [expanded, setExpanded] = useState<string | null>(null);
  const rows = newestFirst(reviews, LIMIT).filter((review) => inGroup(review, group));
  if (!rows.length) return <p className="px-2 py-2 text-xs text-muted-foreground">No traces here yet.</p>;
  return (
    <ol aria-label="Reviewed traces" className="flex flex-col">
      {rows.map((review) => (
        <li
          key={review.execution_id}
          className="motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-top-3 motion-safe:duration-300"
        >
          <DoneRow
            review={review}
            now={now}
            open={expanded === review.execution_id}
            onToggle={() => setExpanded(expanded === review.execution_id ? null : review.execution_id)}
          />
        </li>
      ))}
    </ol>
  );
}
