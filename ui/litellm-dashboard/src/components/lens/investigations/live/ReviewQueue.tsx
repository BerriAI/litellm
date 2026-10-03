import { cn } from "@/lib/cva.config";

import { outcome, reviewKey } from "../../model/live";
import type { Review } from "../../model/types";

const BEFORE = 6;
const AFTER = 12;
const MARK = { issue: "●", clear: "✓", unknown: "?" } as const;

type Row = { review: Review; state: "done" | "active" | "next" };

function rows(played: readonly Review[], current: Review | null, pending: readonly Review[], live: boolean): Row[] {
  const tail = played.slice(live ? -BEFORE : -(BEFORE + AFTER));
  return [
    ...tail.map((review): Row => ({ review, state: "done" })),
    ...(current ? [{ review: current, state: live ? "active" : "done" } as Row] : []),
    ...pending.slice(0, AFTER).map((review): Row => ({ review, state: "next" })),
  ];
}

export function ReviewQueue({
  played,
  current,
  pending,
  live,
}: {
  played: readonly Review[];
  current: Review | null;
  pending: readonly Review[];
  live: boolean;
}) {
  return (
    <ol aria-label="Reviewed traces" className="min-h-0 flex-1 overflow-hidden">
      {rows(played, current, pending, live).map(({ review, state }) => {
        const result = outcome(review);
        return (
          <li
            key={reviewKey(review)}
            data-state={state}
            aria-current={state === "active" ? "step" : undefined}
            className={cn(
              "grid grid-cols-[0.875rem_minmax(0,1fr)_auto] items-center gap-2 border-b border-border/60 px-3.5 py-1.5 text-xs",
              state === "active" && "bg-(--provider)/5 shadow-[inset_2px_0_0_var(--provider)]",
              state === "next" && "text-muted-foreground",
            )}
          >
            {state === "active" && (
              <span
                aria-label="Reviewing"
                className="size-2.5 rounded-full border-[1.5px] border-(--provider)/25 border-t-(--provider) motion-safe:animate-spin"
              />
            )}
            {state === "done" && (
              <span
                aria-label={result}
                className={cn("text-[11px]", result === "issue" ? "text-[#e5484d]" : "text-muted-foreground")}
              >
                {MARK[result]}
              </span>
            )}
            {state === "next" && <span className="text-[11px] text-muted-foreground">·</span>}
            <span className="truncate">{review.agent || review.name}</span>
            <span className="font-mono text-[11px] text-muted-foreground">{review.trace_id.slice(0, 8)}</span>
          </li>
        );
      })}
    </ol>
  );
}
