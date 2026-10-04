"use client";

import { PanelRight, X } from "lucide-react";

import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { agoLabel } from "@/components/view_logs/TraceView/lensField";
import { Button } from "@/components/ui/button";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";

import { money } from "../../model/format";
import { outcome, providerOf, queueRows, reviewKey, shortVerdict, type IssueCount, type StripState } from "../../model/live";
import type { Playback } from "../../model/live";
import type { Review } from "../../model/types";

const RECENT = 3;
const MARK = { issue: "●", clear: "✓", unknown: "–" } as const;
const LOGO = { sm: "size-3.5", md: "size-[18px]" } as const;

export function ModelName({ model, size = "sm" }: { model: string; size?: keyof typeof LOGO }) {
  if (!model) return null;
  const provider = providerOf(model);
  return (
    <span data-testid="live-model" className="inline-flex shrink-0 items-center gap-1.5 text-foreground">
      {provider && <ProviderLogo provider={provider} className={cn("shrink-0", LOGO[size])} />}
      <span className={cn("font-mono", size === "md" ? "text-[12px]" : "text-[11px]")}>{model}</span>
    </span>
  );
}

function RecentLine({ review, now, onOpen }: { review: Review; now: number; onOpen: (review: Review) => void }) {
  const result = outcome(review);
  const issue = result === "issue";
  return (
    <li className="motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-top-1 motion-safe:duration-300">
      <button
        type="button"
        onClick={() => onOpen(review)}
        className="grid w-full grid-cols-[0.75rem_minmax(0,9rem)_minmax(0,1fr)_auto] items-center gap-2 rounded px-1.5 py-0.5 text-left hover:bg-muted"
      >
        <span aria-label={result} className={issue ? "text-[#e5484d]" : ""}>
          {MARK[result]}
        </span>
        <span className="truncate text-foreground">{review.agent || review.name}</span>
        <span className={cn("truncate", issue && "text-[#e5484d]")}>{shortVerdict(review)}</span>
        <span className="tabular-nums">{agoLabel(Date.parse(review.at), now)}</span>
      </button>
    </li>
  );
}

function issueLabel({ count, scope }: IssueCount): string {
  const noun = count === 1 ? "issue" : "issues";
  if (scope === "findings") return `${count} ${noun} found`;
  return scope ? `${count} ${noun} ${scope}` : `${count} ${noun}`;
}

export function LiveStrip({
  model,
  state,
  playback,
  reviewed,
  selected,
  issues,
  cost,
  onOpen,
  onClose,
}: {
  model: string;
  state: StripState;
  playback: Pick<Playback, "played" | "current">;
  reviewed: number;
  selected: number;
  issues: IssueCount;
  cost: number;
  onOpen: (review?: Review) => void;
  onClose: () => void;
}) {
  const now = useNow(5000);
  const recent = queueRows(playback, RECENT);
  return (
    <section
      aria-label="Live trace results"
      className="flex flex-wrap items-start gap-x-6 gap-y-2 rounded-md bg-muted/30 px-3 py-2 text-[12px] text-muted-foreground"
    >
      <div className="flex min-w-[16rem] flex-col gap-1">
        <ModelName model={model} />
        <span className="tabular-nums">
          {reviewed} of {selected} traces · {issueLabel(issues)} · {money(cost)}
        </span>
      </div>
      <div className="min-w-0 flex-1">
        {state.kind === "failed" && (
          <p role="alert" className="py-0.5 text-[#e5484d]">
            {state.message}
          </p>
        )}
        {state.kind === "waiting" && (
          <p role="status" className="py-0.5">
            {state.message}
          </p>
        )}
        {recent.length > 0 && state.kind !== "waiting" && (
          <ol aria-label="Recently reviewed traces" className="flex flex-col">
            {recent.map((review) => (
              <RecentLine key={reviewKey(review)} review={review} now={now} onOpen={onOpen} />
            ))}
          </ol>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-1">
        <Button variant="outline" size="xs" onClick={() => onOpen()} disabled={!recent.length}>
          <PanelRight />
          View run
        </Button>
        <Button variant="ghost" size="icon-xs" aria-label="Hide live trace results" onClick={onClose}>
          <X />
        </Button>
      </div>
    </section>
  );
}
