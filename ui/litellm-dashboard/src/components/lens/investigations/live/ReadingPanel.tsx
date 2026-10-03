import type { Ref } from "react";

import { cn } from "@/lib/cva.config";

import { modelName, outcome, verdictLine, type Phase } from "../../model/live";
import type { Review } from "../../model/types";
import { ProviderLogo } from "./ProviderBadge";

const VERDICT_TONE = {
  issue: "border-[#e5484d]/30 bg-[#e5484d]/5 text-[#e5484d]",
  clear: "border-border bg-muted/60 text-foreground",
  unknown: "border-border bg-muted/60 text-muted-foreground",
} as const;
const VERDICT_MARK = { issue: "●", clear: "✓", unknown: "?" } as const;

function Spans({ review, phase }: { review: Review; phase: Phase }) {
  return (
    <section aria-label="Trace" className="overflow-hidden rounded-xl border">
      <header className="flex justify-between gap-3 border-b bg-muted/40 px-3 py-2 font-mono text-[11px] text-muted-foreground">
        <span className="truncate">trace {review.trace_id.slice(0, 8)}</span>
        <span className="truncate">{review.agent || review.name}</span>
      </header>
      <ol>
        {review.spans.map((span, index) => {
          const scanning = phase.span === index;
          const cited = phase.verdict && span.cited;
          return (
            <li
              key={span.span_id}
              data-state={scanning ? "reading" : cited ? "cited" : "idle"}
              className={cn(
                "grid grid-cols-[4.5rem_minmax(0,1fr)] gap-2.5 border-b border-dashed border-border/70 px-3 py-1.5 text-[12.5px] transition-colors duration-200 last:border-b-0 motion-reduce:transition-none",
                scanning && "bg-(--provider)/8",
                cited && "bg-[#e5484d]/5",
              )}
            >
              <span className="truncate pt-px font-mono text-[10.5px] text-muted-foreground" title={span.name}>
                {span.kind || span.name}
              </span>
              <span className={cn("line-clamp-2 break-words", cited && "text-[#e5484d]")}>
                {span.preview || span.name}
              </span>
            </li>
          );
        })}
        {!review.spans.length && <li className="px-3 py-2 text-xs text-muted-foreground">No trace parts shown</li>}
      </ol>
    </section>
  );
}

function Reasoning({ review, phase, tokens }: { review: Review; phase: Phase; tokens: string }) {
  const typing = phase.typed < review.reasoning.length;
  return (
    <section aria-label="Reasoning" className="rounded-xl border border-(--provider)/25 bg-(--provider)/3">
      <header className="flex items-center justify-between gap-3 border-b border-(--provider)/20 px-3 py-2">
        <span className="inline-flex min-w-0 items-center gap-1.5 text-xs font-semibold text-(--provider)">
          <ProviderLogo model={review.model} />
          <span className="truncate">{modelName(review.model) || "analysis model"} · reasoning</span>
        </span>
        <span className="shrink-0 font-mono text-[11px] text-muted-foreground">{tokens}</span>
      </header>
      <p className="min-h-24 px-3 py-2.5 text-[12.5px] leading-relaxed whitespace-pre-wrap text-foreground/80">
        {review.reasoning ? review.reasoning.slice(0, phase.typed) : !typing && "No reasoning was returned."}
        {typing && (
          <span
            aria-hidden="true"
            className="ml-px inline-block h-3.5 w-1.5 translate-y-0.5 bg-(--provider) motion-safe:animate-pulse"
          />
        )}
      </p>
    </section>
  );
}

export function ReadingPanel({
  review,
  phase,
  tokens,
  verdictRef,
}: {
  review: Review | null;
  phase: Phase;
  tokens: string;
  verdictRef: Ref<HTMLDivElement>;
}) {
  if (!review) {
    return (
      <p className="px-4 py-6 text-sm text-muted-foreground">
        Traces appear here as the worker reviews them.
      </p>
    );
  }
  const result = outcome(review);
  return (
    <div className="flex min-h-0 flex-col gap-3 overflow-hidden px-4 py-3.5">
      <Spans review={review} phase={phase} />
      <Reasoning review={review} phase={phase} tokens={tokens} />
      <div
        ref={verdictRef}
        role="status"
        aria-label="Verdict"
        data-outcome={result}
        className={cn(
          "flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-sm font-semibold transition-all duration-200 motion-reduce:transition-none",
          VERDICT_TONE[result],
          phase.verdict ? "translate-y-0 opacity-100" : "translate-y-1 opacity-0",
        )}
      >
        <span aria-hidden="true">{VERDICT_MARK[result]}</span>
        <span className="min-w-0 truncate">{phase.verdict ? verdictLine(review) : ""}</span>
      </div>
    </div>
  );
}
