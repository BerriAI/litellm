"use client";

import { useLayoutEffect, useRef, useState } from "react";

import { agoLabel } from "@/components/view_logs/TraceView/lensField";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";

import { briefReasoning, durationLabel, inGroup, liveRows, outcome, shortVerdict, type InFlight } from "../../model/live";
import type { Review } from "../../model/types";
import { ModelName } from "./LiveStrip";

const LIMIT = 200;
const RED = "text-[#e5484d]";
const ROW = "grid h-9 w-full grid-cols-[0.75rem_minmax(0,8rem)_minmax(0,1fr)_auto] items-center gap-2 px-2 text-left text-[12px]";

export type LensMode = "live" | "settling" | "off";

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

function ReadingRow({ item, now }: { item: InFlight; now: number }) {
  return (
    <div className={ROW}>
      <span aria-label="reading" className="size-1.5 rounded-full bg-foreground motion-safe:animate-pulse" />
      <span className="truncate text-foreground">{item.agent || "trace"}</span>
      <span className="truncate font-mono text-[11px] text-muted-foreground">{item.trace_id.slice(0, 8)}</span>
      <span className="text-[11px] tabular-nums text-muted-foreground">
        reading · {durationLabel(Math.max(0, now - Date.parse(item.started_at)))}
      </span>
    </div>
  );
}

function DoneRow({ review, now, open, onToggle }: { review: Review; now: number; open: boolean; onToggle: () => void }) {
  const result = outcome(review);
  return (
    <>
      <button type="button" aria-expanded={open} onClick={onToggle} className={cn(ROW, "rounded-lg hover:bg-muted/50")}>
        <span
          aria-label={result}
          className={cn("size-1.5 rounded-full", result === "issue" ? "bg-[#e5484d]" : "bg-muted-foreground/40")}
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
        <span className="text-[11px] tabular-nums text-muted-foreground">
          {durationLabel(review.duration_ms)} · {agoLabel(Date.parse(review.at), now)}
        </span>
      </button>
      {open && <Expanded review={review} />}
    </>
  );
}

function useLensBox(target: string, layout: string) {
  const list = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState<{ top: number; height: number } | null>(null);
  useLayoutEffect(() => {
    const nodes = [...(list.current?.querySelectorAll<HTMLElement>(`[data-lens="${target}"]`) ?? [])];
    const first = nodes[0];
    const last = nodes.at(-1);
    setBox(first && last ? { top: first.offsetTop, height: last.offsetTop + last.offsetHeight - first.offsetTop } : null);
  }, [target, layout]);
  return { list, box };
}

function Lens({ box, model, mode }: { box: { top: number; height: number }; model: string; mode: LensMode }) {
  return (
    <div
      aria-hidden="true"
      className={cn(
        "pointer-events-none absolute inset-x-0 top-0 rounded-xl bg-muted ring-1 ring-border",
        "transition-[transform,height,opacity] duration-[250ms] ease-out motion-reduce:transition-none",
        mode === "settling" && "opacity-0 duration-[400ms]",
      )}
      style={{ transform: `translateY(${box.top}px)`, height: box.height }}
    >
      <span className="absolute -top-2.5 right-3 inline-flex items-center rounded-full bg-background px-2 py-0.5 shadow-sm ring-1 ring-border">
        <ModelName model={model} size="md" />
      </span>
    </div>
  );
}

export function TraceList({
  reading,
  reviews,
  model,
  group,
  mode,
  nowLine,
}: {
  reading: readonly InFlight[];
  reviews: readonly Review[];
  model: string;
  group: string | null;
  mode: LensMode;
  nowLine: string | null;
}) {
  const now = useNow(reading.length ? 200 : 5000);
  const [expanded, setExpanded] = useState<string | null>(null);
  const rows = liveRows(reading, reviews, LIMIT).filter((row) => row.kind === "reading" || inGroup(row.review, group));
  const inFlight = rows.some((row) => row.kind === "reading");
  const target = inFlight ? "reading" : "slot";
  const layout = `${expanded}|${rows.map((row) => `${row.kind}:${row.key}`).join(",")}`;
  const { list, box } = useLensBox(target, layout);
  const showSlot = mode === "live" && !inFlight;
  return (
    <div ref={list} className="relative pt-3">
      {box && mode !== "off" && <Lens box={box} model={model} mode={mode} />}
      {showSlot && nowLine && (
        <div data-lens="slot" role="status" className={cn(ROW, "relative text-muted-foreground")}>
          <span aria-hidden="true" className="size-1.5 rounded-full bg-foreground motion-safe:animate-pulse" />
          <span className="col-span-3 truncate tabular-nums">{nowLine}</span>
        </div>
      )}
      <ol aria-label="Reviewed traces" className="relative flex flex-col">
        {rows.map((row) => (
          <li key={row.key} data-lens={row.kind === "reading" ? "reading" : undefined}>
            {row.kind === "reading" ? (
              <ReadingRow item={row.item} now={now} />
            ) : (
              <DoneRow
                review={row.review}
                now={now}
                open={expanded === row.key}
                onToggle={() => setExpanded(expanded === row.key ? null : row.key)}
              />
            )}
          </li>
        ))}
      </ol>
      {!rows.length && !showSlot && (
        <p className="px-2 py-2 text-[12px] text-muted-foreground">No traces in this group yet.</p>
      )}
    </div>
  );
}
