import { cn } from "@/lib/cva.config";

import { outcome, verdictLine, type Phase } from "../../model/live";
import { spanPreviewLines } from "../../model/spanPreview";
import type { Review } from "../../model/types";

const VERDICT_TONE = {
  issue: "text-[#e5484d]",
  clear: "text-foreground",
  unknown: "text-muted-foreground",
} as const;
const VERDICT_DOT = {
  issue: "bg-[#e5484d]",
  clear: "bg-muted-foreground/40",
  unknown: "bg-muted-foreground/40",
} as const;

function Spans({ review, phase }: { review: Review; phase: Phase }) {
  return (
    <ol aria-label="Trace" className="divide-y divide-border/60 border-y border-border/60">
      {review.spans.map((span, index) => {
        const reading = phase.span === index;
        const cited = phase.verdict && span.cited;
        return (
          <li
            key={span.span_id}
            data-state={reading ? "reading" : cited ? "cited" : "idle"}
            className={cn(
              "px-4 py-2 text-[12px] transition-colors duration-150 motion-reduce:transition-none",
              reading && "bg-trace-row-hover",
              cited && "shadow-[inset_2px_0_0_#e5484d]",
            )}
          >
            <p className="mb-0.5 flex gap-2 font-mono text-[11px] text-muted-foreground">
              <span>{span.kind}</span>
              <span className="truncate">{span.name}</span>
            </p>
            {spanPreviewLines(span.preview).map((line, n) => (
              <p
                key={n}
                className={cn("line-clamp-3 break-words", line.error ? "text-[#e5484d]" : "text-foreground")}
              >
                {line.label && (
                  <span className="mr-1.5 font-mono text-[11px] text-muted-foreground">{line.label}</span>
                )}
                {line.text}
              </p>
            ))}
          </li>
        );
      })}
      {!review.spans.length && (
        <li className="px-4 py-2 text-[12px] text-muted-foreground">No trace parts were shown to the model</li>
      )}
    </ol>
  );
}

export function ReadingPanel({ review, phase }: { review: Review; phase: Phase }) {
  const result = outcome(review);
  const typing = phase.typed < review.reasoning.length;
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-3 px-4 font-mono text-[11px] text-muted-foreground">
        <span className="truncate text-foreground">{review.agent || review.name}</span>
        <span className="truncate">trace {review.trace_id.slice(0, 8)}</span>
      </div>
      <Spans review={review} phase={phase} />
      <section aria-label="Reasoning" className="mx-4 rounded-md border bg-muted/40">
        <header className="flex h-8 items-center justify-between gap-3 border-b px-3">
          <span className="text-[10px] tracking-[0.08em] text-muted-foreground uppercase">Reasoning</span>
          <span className="truncate font-mono text-[11px] text-muted-foreground">{review.model}</span>
        </header>
        <p className="min-h-12 px-3 py-2 text-[12px] leading-relaxed whitespace-pre-wrap text-foreground">
          {review.reasoning ? review.reasoning.slice(0, phase.typed) : !typing && "No reasoning was returned."}
          {typing && <span aria-hidden="true" className="ml-px inline-block h-3 w-px translate-y-0.5 bg-foreground" />}
        </p>
      </section>
      <div
        role="status"
        aria-label="Verdict"
        data-outcome={result}
        className={cn(
          "mx-4 flex items-center gap-2 text-[12px] font-medium transition-opacity duration-150 motion-reduce:transition-none",
          VERDICT_TONE[result],
          phase.verdict ? "opacity-100" : "opacity-0",
        )}
      >
        <span aria-hidden="true" className={cn("size-1.5 shrink-0 rounded-full", VERDICT_DOT[result])} />
        <span className="min-w-0">{phase.verdict ? verdictLine(review) : ""}</span>
      </div>
    </div>
  );
}
