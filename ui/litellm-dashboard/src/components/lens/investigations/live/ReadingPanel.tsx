import type { Ref } from "react";

import { cn } from "@/lib/cva.config";

import { outcome, verdictLine, type Phase } from "../../model/live";
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
    <table aria-label="Trace" className="w-full table-fixed border-collapse text-left">
      <tbody>
        {review.spans.map((span, index) => {
          const reading = phase.span === index;
          const cited = phase.verdict && span.cited;
          return (
            <tr
              key={span.span_id}
              data-state={reading ? "reading" : cited ? "cited" : "idle"}
              className={cn(
                "h-9 border-b border-border/60 text-[12px] transition-colors duration-150 motion-reduce:transition-none",
                reading && "bg-trace-row-hover",
              )}
            >
              <td className="w-[88px] truncate px-3 font-mono text-[11px] text-muted-foreground" title={span.name}>
                {span.kind || span.name}
              </td>
              <td className={cn("truncate px-3", cited ? "text-[#e5484d]" : "text-foreground")} title={span.preview}>
                {span.preview || span.name}
              </td>
            </tr>
          );
        })}
        {!review.spans.length && (
          <tr className="h-9 text-[12px] text-muted-foreground">
            <td className="px-3">No trace parts were shown to the model</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

export function ReadingPanel({
  review,
  phase,
  verdictRef,
}: {
  review: Review | null;
  phase: Phase;
  verdictRef: Ref<HTMLDivElement>;
}) {
  if (!review) {
    return <p className="px-3 py-6 text-[12px] text-muted-foreground">Waiting for the first trace review.</p>;
  }
  const result = outcome(review);
  const typing = phase.typed < review.reasoning.length;
  return (
    <div className="flex flex-col">
      <div className="flex h-8 items-center justify-between gap-3 border-b border-border px-3 font-mono text-[11px] text-muted-foreground">
        <span className="truncate">trace {review.trace_id.slice(0, 8)}</span>
        <span className="truncate">{review.agent || review.name}</span>
      </div>
      <Spans review={review} phase={phase} />
      <section aria-label="Reasoning" className="m-3 rounded-md border bg-muted/40">
        <header className="flex h-8 items-center justify-between gap-3 border-b px-3">
          <span className="text-[10px] tracking-[0.08em] text-muted-foreground uppercase">Reasoning</span>
          <span className="truncate font-mono text-[11px] text-muted-foreground">{review.model}</span>
        </header>
        <p className="min-h-16 px-3 py-2 text-[12px] leading-relaxed whitespace-pre-wrap text-foreground">
          {review.reasoning ? review.reasoning.slice(0, phase.typed) : !typing && "No reasoning was returned."}
          {typing && <span aria-hidden="true" className="ml-px inline-block h-3 w-px translate-y-0.5 bg-foreground" />}
        </p>
      </section>
      <div
        ref={verdictRef}
        role="status"
        aria-label="Verdict"
        data-outcome={result}
        className={cn(
          "mx-3 mb-3 flex items-center gap-2 text-[12px] font-medium transition-opacity duration-150 motion-reduce:transition-none",
          VERDICT_TONE[result],
          phase.verdict ? "opacity-100" : "opacity-0",
        )}
      >
        <span aria-hidden="true" className={cn("size-1.5 shrink-0 rounded-full", VERDICT_DOT[result])} />
        <span className="min-w-0 truncate">{phase.verdict ? verdictLine(review) : ""}</span>
      </div>
    </div>
  );
}
