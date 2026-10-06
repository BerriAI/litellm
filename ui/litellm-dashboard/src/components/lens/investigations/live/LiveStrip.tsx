"use client";

import { ArrowRight, PanelRight, X } from "lucide-react";
import type { ReactNode } from "react";

import { useLensApi } from "../../data/LensServices";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { agoLabel } from "../../model/format";
import { Button } from "@/components/ui/button";
import { useNow } from "@/hooks/useNow";
import { cn } from "@/lib/cva.config";

import { money } from "../../model/format";
import {
  newestFirst,
  outcome,
  providerOf,
  reviewKey,
  shortVerdict,
  type IssueCount,
  type StripState,
} from "../../model/live";
import type { Review } from "../../model/types";

const RECENT = 3;
const MARK = { issue: "●", clear: "✓", unknown: "–" } as const;
const LOGO = { sm: "size-3.5", md: "size-[18px]" } as const;

export function ModelName({ model, size = "sm" }: { model: string; size?: keyof typeof LOGO }) {
  const api = useLensApi();
  const { data: catalog } = useModelCostMap(api.scope !== "demo");
  if (!model) return null;
  const provider = providerOf(model, catalog ?? {});
  return (
    <span data-testid="live-model" className="inline-flex shrink-0 items-center gap-1.5 text-foreground">
      {provider && <ProviderLogo provider={provider} className={cn("shrink-0", LOGO[size])} />}
      <span className={cn("font-mono", size === "md" ? "text-xs" : "text-xs")}>{model}</span>
    </span>
  );
}

function RecentLine({ review, now, onOpen }: { review: Review; now: number; onOpen: () => void }) {
  const result = outcome(review);
  const issue = result === "issue";
  return (
    <li className="motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-top-1 motion-safe:duration-300">
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation();
          onOpen();
        }}
        className="grid w-full grid-cols-[0.75rem_minmax(0,9rem)_minmax(0,1fr)_auto] items-center gap-2 rounded px-1.5 py-0.5 text-left hover:bg-muted"
      >
        <span aria-label={result} className={issue ? "text-destructive" : ""}>
          {MARK[result]}
        </span>
        <span className="truncate text-foreground">{review.agent || review.name}</span>
        <span className={cn("truncate", issue && "text-destructive")}>{shortVerdict(review)}</span>
        <span className="tabular-nums">{agoLabel(Date.parse(review.at), now)}</span>
      </button>
    </li>
  );
}

function issueLabel({ count, scope }: IssueCount): string {
  const noun = count === 1 ? "issue" : "issues";
  if (scope === "findings") return `${count} ${noun} found`;
  const flagged = `${count} ${count === 1 ? "trace" : "traces"} flagged`;
  return scope ? `${flagged} ${scope}` : flagged;
}

export function LiveStrip({
  model,
  state,
  reviews,
  reviewed,
  selected,
  reused = 0,
  reusable = 0,
  issues,
  cost,
  waiting,
  onOpen,
  onClose,
}: {
  model: string;
  state: StripState;
  waiting?: ReactNode;
  reviews: readonly Review[];
  reviewed: number;
  selected: number;
  reused?: number;
  reusable?: number;
  issues: IssueCount;
  cost: number;
  onOpen: () => void;
  onClose: () => void;
}) {
  const now = useNow(5000);
  const recent = newestFirst(reviews, RECENT);
  const finished = state.kind === "done" || state.kind === "failed";
  return (
    <section
      aria-label="Live trace results"
      onClick={onOpen}
      className="flex cursor-pointer flex-wrap items-start gap-x-6 gap-y-2 rounded-md bg-muted/30 px-3 py-2 text-xs text-muted-foreground transition-colors hover:bg-muted/50"
    >
      <div className="flex min-w-[16rem] items-start gap-3">
        <div className="flex flex-col gap-1">
          <ModelName model={model} />
          <span className="tabular-nums">
            {reviewed} of {selected} traces · {issueLabel(issues)} · {money(cost)}
          </span>
          {(reused > 0 || reusable > 0) && (
            <span>
              {finished
                ? `${reused} reused without review cost · ${Math.max(0, reviewed - reused)} newly reviewed`
                : `${reusable} eligible for reuse · ${Math.max(0, selected - reusable)} need review`}
            </span>
          )}
        </div>
        <Button
          variant="outline"
          size="xs"
          className="mt-0.5 shrink-0"
          onClick={(event) => {
            event.stopPropagation();
            onOpen();
          }}
        >
          <PanelRight />
          View run
          <ArrowRight />
        </Button>
      </div>
      <div className="min-w-0 flex-1">
        {state.kind === "failed" && (
          <p role="alert" className="py-0.5 text-destructive">
            {state.message}
          </p>
        )}
        {state.kind === "waiting" && (
          <div role="status" className="flex flex-col gap-0.5 py-0.5" onClick={(event) => event.stopPropagation()}>
            {waiting ?? state.message}
          </div>
        )}
        {state.kind === "reviewing" && (
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
      <Button
        variant="ghost"
        size="icon-xs"
        aria-label="Hide live trace results"
        className="shrink-0"
        onClick={(event) => {
          event.stopPropagation();
          onClose();
        }}
      >
        <X />
      </Button>
    </section>
  );
}
