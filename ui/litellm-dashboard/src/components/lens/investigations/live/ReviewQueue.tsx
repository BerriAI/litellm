import { Loader2 } from "lucide-react";

import { agoLabel } from "@/components/view_logs/TraceView/lensField";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";

import { inGroup, outcome, queueRows, reviewKey, shortVerdict, type Playback } from "../../model/live";
import type { Review } from "../../model/types";

const LIMIT = 60;

export function ReviewQueue({
  playback,
  live,
  focused,
  group,
  onPick,
}: {
  playback: Pick<Playback, "played" | "current">;
  live: boolean;
  focused: Review | null;
  group: string | null;
  onPick: (review: Review) => void;
}) {
  const now = useNow(5000);
  const rows = queueRows(playback, LIMIT).filter((review) => inGroup(review, group));
  if (!rows.length) return <p className="py-2 text-[12px] text-muted-foreground">No traces in this group yet.</p>;
  return (
    <ol aria-label="Reviewed traces" className="flex flex-col">
      {rows.map((review) => {
        const reading = live && review === playback.current;
        const result = outcome(review);
        const selected = review === focused;
        return (
          <li key={reviewKey(review)} className={reading ? "motion-safe:animate-in motion-safe:fade-in" : ""}>
            <button
              type="button"
              aria-current={selected ? "true" : undefined}
              onClick={() => onPick(review)}
              className={cn(
                "grid w-full grid-cols-[0.75rem_minmax(0,7.5rem)_minmax(0,1fr)_auto] items-center gap-2 rounded-md px-2 py-1.5 text-left text-[12px] hover:bg-muted",
                selected && "bg-background ring-[1.5px] ring-inset ring-foreground",
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
              <span className={cn("truncate", result === "issue" ? "text-[#e5484d]" : "text-muted-foreground")}>
                {reading ? "reading…" : shortVerdict(review)}
              </span>
              <span className="text-[11px] tabular-nums text-muted-foreground">
                {agoLabel(Date.parse(review.at), now)}
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
