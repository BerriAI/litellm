import { Loader2 } from "lucide-react";

import { cn } from "@/lib/cva.config";

import { outcome, queueRows, reviewKey, type Playback } from "../../model/live";

const LIMIT = 80;

export function ReviewQueue({ playback, live }: { playback: Pick<Playback, "played" | "current">; live: boolean }) {
  return (
    <table aria-label="Reviewed traces" className="w-full table-fixed border-collapse text-left">
      <tbody>
        {queueRows(playback, LIMIT).map((review) => {
          const active = live && review === playback.current;
          const result = outcome(review);
          return (
            <tr
              key={reviewKey(review)}
              data-state={active ? "active" : result}
              aria-current={active ? "step" : undefined}
              className={cn(
                "h-9 border-b border-border/60 text-[12px]",
                active && "bg-trace-row-hover",
                review === playback.current && "motion-safe:animate-in motion-safe:fade-in",
              )}
            >
              <td className="w-7 pl-3">
                {active ? (
                  <Loader2 aria-label="Reviewing" className="size-3 text-muted-foreground motion-safe:animate-spin" />
                ) : (
                  <span
                    aria-label={result}
                    className={cn(
                      "block size-1.5 rounded-full",
                      result === "issue" ? "bg-[#e5484d]" : "bg-muted-foreground/40",
                    )}
                  />
                )}
              </td>
              <td className="truncate px-2 text-foreground">{review.agent || review.name}</td>
              <td className="w-[84px] truncate px-3 text-right font-mono text-[11px] text-muted-foreground">
                {review.trace_id.slice(0, 8)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
