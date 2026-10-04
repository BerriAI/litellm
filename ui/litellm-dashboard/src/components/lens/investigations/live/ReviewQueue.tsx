import { Loader2 } from "lucide-react";

import { cn } from "@/lib/cva.config";

import { outcome, queueRows, reviewKey, type Playback } from "../../model/live";
import type { Review } from "../../model/types";

const LIMIT = 60;

export function ReviewQueue({
  playback,
  live,
  focused,
  onPick,
}: {
  playback: Pick<Playback, "played" | "current">;
  live: boolean;
  focused: Review | null;
  onPick: (review: Review) => void;
}) {
  return (
    <ol aria-label="Reviewed traces" className="divide-y divide-border/60">
      {queueRows(playback, LIMIT).map((review) => {
        const reading = live && review === playback.current;
        const result = outcome(review);
        const selected = review === focused;
        return (
          <li key={reviewKey(review)} className={review === playback.current ? "motion-safe:animate-in motion-safe:fade-in" : ""}>
            <button
              type="button"
              aria-current={selected ? "true" : undefined}
              onClick={() => onPick(review)}
              className={cn(
                "grid h-8 w-full grid-cols-[1.25rem_minmax(0,1fr)_auto] items-center gap-2 px-4 text-left text-[12px] hover:bg-trace-row-hover",
                selected && "bg-trace-row-hover",
              )}
            >
              {reading ? (
                <Loader2 aria-label="Reviewing" className="size-3 text-muted-foreground motion-safe:animate-spin" />
              ) : (
                <span
                  aria-label={result}
                  className={cn("size-1.5 rounded-full", result === "issue" ? "bg-[#e5484d]" : "bg-muted-foreground/40")}
                />
              )}
              <span className="truncate text-foreground">{review.agent || review.name}</span>
              <span className="font-mono text-[11px] text-muted-foreground">{review.trace_id.slice(0, 8)}</span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
