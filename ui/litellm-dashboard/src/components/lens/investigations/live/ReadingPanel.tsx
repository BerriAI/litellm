import { Wrench } from "lucide-react";

import { cn } from "@/lib/cva.config";

import { outcome, type Outcome, type Phase } from "../../model/live";
import { timeline, type TimelineItem } from "../../model/spanPreview";
import type { Review } from "../../model/types";
import { ModelName } from "./LiveStrip";

const RED = "text-[#e5484d]";
const CHIP: Record<Outcome, { label: string; className: string }> = {
  issue: { label: "Issue", className: "bg-[#e5484d]/10 text-[#e5484d] ring-1 ring-inset ring-[#e5484d]/30" },
  clear: { label: "Looks fine", className: "bg-muted text-foreground" },
  unknown: { label: "Not enough evidence", className: "bg-muted/60 text-muted-foreground" },
};
const PANE_LABEL = "text-[11px] font-medium text-muted-foreground";

function VerdictChip({ result, decided }: { result: Outcome; decided: boolean }) {
  if (!decided) {
    return <span className="rounded-full bg-muted/60 px-2 py-0.5 text-[11px] text-muted-foreground">Reading…</span>;
  }
  const chip = CHIP[result];
  return (
    <span
      data-testid="verdict-chip"
      className={cn(
        "rounded-full px-2 py-0.5 text-[11px] font-medium motion-safe:animate-in motion-safe:zoom-in-95 motion-safe:fade-in",
        chip.className,
      )}
    >
      {chip.label}
    </span>
  );
}

function Bubble({ who, text, tone }: { who: string; text: string; tone: "ask" | "reply" }) {
  return (
    <div className={cn("flex flex-col gap-1", tone === "reply" && "items-end")}>
      <span className="text-[11px] text-muted-foreground">{who}</span>
      <p
        className={cn(
          "max-w-[92%] rounded-2xl px-3 py-2 text-[13px] leading-relaxed break-words whitespace-pre-wrap text-foreground",
          tone === "ask" ? "rounded-tl-sm bg-muted/70" : "rounded-tr-sm border bg-background",
        )}
      >
        {text}
      </p>
    </div>
  );
}

function ToolRow({ item }: { item: Extract<TimelineItem, { kind: "tool" }> }) {
  return (
    <div className="flex gap-2 text-[12px]">
      <Wrench aria-hidden="true" className={cn("mt-0.5 size-3.5 shrink-0 text-muted-foreground", item.error && RED)} />
      <div className="min-w-0 flex-1">
        <p className="flex min-w-0 gap-1.5">
          <span className="shrink-0 font-medium text-foreground">{item.name}</span>
          <span className="truncate text-muted-foreground">{item.args}</span>
        </p>
        {item.result && (
          <p className={cn("line-clamp-2 break-words", item.error ? RED : "text-muted-foreground")}>
            <span aria-hidden="true">↳ </span>
            {item.result}
          </p>
        )}
      </div>
    </div>
  );
}

function Step({ item }: { item: TimelineItem }) {
  switch (item.kind) {
    case "ask":
      return <Bubble who="User asked" text={item.text} tone="ask" />;
    case "reply":
      return <Bubble who="Agent replied" text={item.text} tone="reply" />;
    case "tool":
      return <ToolRow item={item} />;
    case "failure":
      return <p className={cn("text-[12px]", RED)}>{item.text}</p>;
    case "note":
      return (
        <p className="line-clamp-2 text-[12px] text-muted-foreground">
          <span className="mr-1.5 font-medium text-foreground">{item.label}</span>
          {item.text}
        </p>
      );
  }
}

function WhatHappened({ review, items, phase }: { review: Review; items: readonly TimelineItem[]; phase: Phase }) {
  if (!items.length) return <p className="text-[12px] text-muted-foreground">No trace parts were shown to the model</p>;
  return (
    <ol aria-label="What happened" className="flex flex-col gap-1">
      {items.map((item, index) => {
        const reading = phase.span === index;
        const cited = phase.verdict && !!review.spans[item.span]?.cited;
        return (
          <li
            key={`${item.kind}-${item.span}-${index}`}
            data-state={reading ? "reading" : cited ? "cited" : "idle"}
            className={cn(
              "rounded-lg px-2.5 py-2 transition-colors duration-150 motion-reduce:transition-none",
              reading && "bg-trace-row-hover ring-1 ring-inset ring-foreground/15",
              cited && "ring-[1.5px] ring-inset ring-[#e5484d]",
            )}
          >
            <Step item={item} />
          </li>
        );
      })}
    </ol>
  );
}

export function ReadingPanel({ review, phase }: { review: Review; phase: Phase }) {
  const result = outcome(review);
  const items = timeline(review.spans);
  const typing = phase.typed < review.reasoning.length;
  const verdicts = review.verdicts.length
    ? review.verdicts
    : [{ check_id: "", kind: "pattern" as const, summary: review.cannot_assess ? "Not enough evidence to judge" : "No issue observed" }];
  return (
    <div className="flex flex-col gap-5">
      <header className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="truncate text-[14px] font-semibold text-foreground">{review.agent || review.name}</h3>
          <p className="truncate font-mono text-[11px] text-muted-foreground">trace {review.trace_id}</p>
        </div>
        <VerdictChip result={result} decided={phase.verdict} />
      </header>
      <section className="flex flex-col gap-2">
        <h4 className={PANE_LABEL}>What happened</h4>
        <WhatHappened review={review} items={items} phase={phase} />
      </section>
      <section aria-label="Reasoning" className="flex flex-col gap-2">
        <h4 className={cn(PANE_LABEL, "flex items-center gap-2")}>
          <span>Lens&apos;s reasoning</span>
          <ModelName model={review.model} />
        </h4>
        <p className="min-h-10 text-[13px] leading-relaxed whitespace-pre-wrap text-foreground">
          {review.reasoning ? review.reasoning.slice(0, phase.typed) : !typing && "No reasoning was returned."}
          {typing && <span aria-hidden="true" className="ml-px inline-block h-3.5 w-px translate-y-0.5 bg-foreground" />}
        </p>
      </section>
      <ul
        role="status"
        aria-label="Verdict"
        data-outcome={result}
        className={cn(
          "flex flex-col gap-1.5 transition-opacity duration-150 motion-reduce:transition-none",
          phase.verdict ? "opacity-100" : "opacity-0",
        )}
      >
        {phase.verdict &&
          verdicts.map((verdict, index) => (
            <li
              key={`${verdict.check_id}-${index}`}
              className={cn(
                "flex gap-2 text-[13px] leading-snug",
                verdict.kind === "issue" ? RED : result === "unknown" ? "text-muted-foreground" : "text-foreground",
              )}
            >
              <span
                aria-hidden="true"
                className={cn(
                  "mt-1.5 size-1.5 shrink-0 rounded-full",
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
