"use client";

import { useLayoutEffect, useRef, useState } from "react";

import { agoLabel } from "@/components/view_logs/TraceView/lensField";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";

import { briefReasoning, inGroup, outcome, reviewKey, shortVerdict, traceRows, type Playback } from "../../model/live";
import type { Review } from "../../model/types";
import { ModelName } from "./LiveStrip";

const LIMIT = 80;
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

function useHighlighter(active: string | null, layout: string) {
  const list = useRef<HTMLOListElement>(null);
  const [box, setBox] = useState<{ top: number; height: number } | null>(null);
  useLayoutEffect(() => {
    const row = active ? list.current?.querySelector<HTMLElement>(`[data-row="${CSS.escape(active)}"]`) : null;
    setBox(row ? { top: row.offsetTop, height: row.offsetHeight } : null);
  }, [active, layout]);
  return { list, box };
}

export function TraceList({
  playback,
  model,
  live,
  group,
}: {
  playback: Pick<Playback, "played" | "current" | "pending">;
  model: string;
  live: boolean;
  group: string | null;
}) {
  const now = useNow(5000);
  const [expanded, setExpanded] = useState<string | null>(null);
  const rows = traceRows(playback, LIMIT)
    .map((row) => (live || row.state !== "reviewing" ? row : { ...row, state: "done" as const }))
    .filter(({ review, state }) => state !== "done" || inGroup(review, group));
  const active = live && playback.current ? reviewKey(playback.current) : null;
  const { list, box } = useHighlighter(active, `${expanded}|${rows.map((row) => reviewKey(row.review)).join(",")}`);
  if (!rows.length) return <p className="py-2 text-[12px] text-muted-foreground">No traces yet.</p>;
  return (
    <ol aria-label="Reviewed traces" ref={list} className="relative flex flex-col">
      {box && (
        <li
          aria-hidden="true"
          className="pointer-events-none absolute inset-x-0 top-0 rounded-lg bg-muted ring-1 ring-border transition-[transform,height] duration-200 ease-out motion-reduce:transition-none"
          style={{ transform: `translateY(${box.top}px)`, height: box.height }}
        />
      )}
      {rows.map(({ review, state }) => {
        const key = reviewKey(review);
        const result = outcome(review);
        const open = expanded === key;
        const reviewing = state === "reviewing";
        const queued = state === "queued";
        return (
          <li key={key} data-row={key} className="relative">
            <button
              type="button"
              aria-expanded={open}
              disabled={queued}
              onClick={() => setExpanded(open ? null : key)}
              className={cn(
                "grid w-full grid-cols-[0.75rem_minmax(0,8rem)_minmax(0,1fr)_auto] items-center gap-2 rounded-lg px-2 py-2 text-left text-[12px]",
                !reviewing && !queued && "hover:bg-muted/50",
                queued && "opacity-40",
              )}
            >
              <span
                aria-label={queued ? "queued" : reviewing ? "reviewing" : result}
                className={cn(
                  "size-1.5 rounded-full",
                  reviewing
                    ? "bg-foreground motion-safe:animate-pulse"
                    : result === "issue" && !queued
                      ? "bg-[#e5484d]"
                      : "bg-muted-foreground/40",
                )}
              />
              <span className="truncate text-foreground">{review.agent || review.name}</span>
              {reviewing ? (
                <span className="flex min-w-0 items-center gap-2 text-muted-foreground">
                  <span className="inline-flex shrink-0 items-center rounded-md bg-background px-1.5 py-0.5 ring-1 ring-border">
                    <ModelName model={model} />
                  </span>
                  <span className="truncate">reading…</span>
                </span>
              ) : (
                <span className={cn("truncate", result === "issue" && !queued ? RED : "text-muted-foreground")}>
                  {queued ? "up next" : shortVerdict(review)}
                </span>
              )}
              <span className="text-[11px] tabular-nums text-muted-foreground">
                {agoLabel(Date.parse(review.at), now)}
              </span>
            </button>
            {open && <Expanded review={review} />}
          </li>
        );
      })}
    </ol>
  );
}
