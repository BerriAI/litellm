"use client";

import { useState } from "react";

import { agoLabel } from "@/components/view_logs/TraceView/lensField";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";

import { briefReasoning, durationLabel, inGroup, liveRows, outcome, shortVerdict, type InFlight } from "../../model/live";
import type { Review } from "../../model/types";
import { ModelName } from "./LiveStrip";

const LIMIT = 200;
const RED = "text-[#e5484d]";

function Expanded({ review }: { review: Review }) {
  const verdicts = review.verdicts.length
    ? review.verdicts
    : [{ check_id: "", kind: "pattern" as const, summary: review.cannot_assess ? "Not enough evidence to judge" : "No issue observed" }];
  return (
    <div className="flex flex-col gap-1.5 px-7 pt-0.5 pb-2.5 text-[12px] leading-relaxed">
      {review.reasoning && <p className="text-muted-foreground">{briefReasoning(review.reasoning)}</p>}
      <ul className="flex flex-col gap-1">
        {verdicts.map((verdict, index) => (
          <li key={`${verdict.check_id}-${index}`} className={cn("flex gap-2", verdict.kind === "issue" ? RED : "text-foreground")}>
            <span
              aria-hidden="true"
              className={cn(
                "mt-[0.45rem] size-1.5 shrink-0 rounded-full",
                verdict.kind === "issue" ? "bg-[#e5484d]" : "bg-muted-foreground/40",
              )}
            />
            <span className="min-w-0">{verdict.summary}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ReadingRow({ item, model, now }: { item: InFlight; model: string; now: number }) {
  return (
    <div className="grid w-full grid-cols-[0.75rem_minmax(0,8rem)_minmax(0,1fr)_auto] items-center gap-2 rounded-lg bg-muted px-2 py-2 text-[12px] ring-1 ring-border transition-colors duration-300">
      <span aria-label="reading" className="size-1.5 rounded-full bg-foreground motion-safe:animate-pulse" />
      <span className="truncate text-foreground">{item.agent || "trace"}</span>
      <span className="flex min-w-0 items-center gap-2 text-muted-foreground">
        <span className="inline-flex shrink-0 items-center rounded-md bg-background px-1.5 py-0.5 ring-1 ring-border">
          <ModelName model={model} />
        </span>
        <span className="truncate font-mono text-[11px]">{item.trace_id.slice(0, 8)}</span>
      </span>
      <span className="text-[11px] tabular-nums text-muted-foreground">
        reading · {durationLabel(Math.max(0, now - Date.parse(item.started_at)))}
      </span>
    </div>
  );
}

export function TraceList({
  reading,
  reviews,
  model,
  group,
}: {
  reading: readonly InFlight[];
  reviews: readonly Review[];
  model: string;
  group: string | null;
}) {
  const now = useNow(reading.length ? 200 : 5000);
  const [expanded, setExpanded] = useState<string | null>(null);
  const rows = liveRows(reading, reviews, LIMIT).filter((row) => row.kind === "reading" || inGroup(row.review, group));
  if (!rows.length) return <p className="px-2 py-2 text-[12px] text-muted-foreground">No traces in this group yet.</p>;
  return (
    <ol aria-label="Reviewed traces" className="flex flex-col">
      {rows.map((row) => {
        if (row.kind === "reading") {
          return (
            <li key={row.key} className="motion-safe:animate-in motion-safe:fade-in motion-safe:duration-150">
              <ReadingRow item={row.item} model={model} now={now} />
            </li>
          );
        }
        const { review, key } = row;
        const result = outcome(review);
        const open = expanded === key;
        return (
          <li
            key={key}
            className="motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-top-1 motion-safe:duration-150"
          >
            <button
              type="button"
              aria-expanded={open}
              onClick={() => setExpanded(open ? null : key)}
              className="grid w-full grid-cols-[0.75rem_minmax(0,8rem)_minmax(0,1fr)_auto] items-center gap-2 rounded-lg px-2 py-2 text-left text-[12px] transition-colors duration-300 hover:bg-muted/50"
            >
              <span
                aria-label={result}
                className={cn("size-1.5 rounded-full", result === "issue" ? "bg-[#e5484d]" : "bg-muted-foreground/40")}
              />
              <span className="truncate text-foreground">{review.agent || review.name}</span>
              <span className={cn("truncate", result === "issue" ? RED : "text-muted-foreground")}>
                {shortVerdict(review)}
              </span>
              <span className="text-[11px] tabular-nums text-muted-foreground">
                {durationLabel(review.duration_ms)} · {agoLabel(Date.parse(review.at), now)}
              </span>
            </button>
            {open && <Expanded review={review} />}
          </li>
        );
      })}
    </ol>
  );
}
