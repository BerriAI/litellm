"use client";

import { useQuery } from "@tanstack/react-query";
import { MessageSquareQuote, Star } from "lucide-react";

import { cn } from "@/lib/cva.config";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";

import { useTracesApi } from "../../api";
import { formatScore } from "../../list/AgentTracesTable";
import { LOW_SCORE } from "../../list/useTraceFeedback";
import type { Feedback, TraceSummary } from "../../types";
import { feedbackView } from "./feedback";

interface FeedbackPanelProps {
  summary: Pick<TraceSummary, "trace_id" | "trace_ref">;
  accessToken: string;
}

const isLow = (score: number) => score <= LOW_SCORE;

function Entry({ entry }: { entry: Feedback }) {
  const low = isLow(entry.score);
  return (
    <li className="flex gap-3 py-2 first:pt-0 last:pb-0" data-testid="feedback-entry" data-low={low || undefined}>
      <span
        className={cn(
          "flex h-9 w-12 shrink-0 items-baseline justify-center rounded-md pt-1.5 font-mono tabular-nums",
          low ? "bg-destructive/10 text-destructive" : "bg-muted text-foreground",
        )}
      >
        <span className="text-base font-semibold">{entry.score}</span>
        <span className="text-xs opacity-60">/10</span>
      </span>
      <div className="flex min-w-0 flex-col gap-0.5">
        {entry.comment ? (
          <p className="text-sm whitespace-pre-wrap text-foreground">“{entry.comment}”</p>
        ) : (
          <p className="text-sm text-muted-foreground italic">No comment</p>
        )}
        <span className="text-xs text-muted-foreground">
          {entry.author} · <span title={entry.updated_at}>{formatActivityTimestamp(entry.updated_at)}</span>
        </span>
      </div>
    </li>
  );
}

export function FeedbackPanel({ summary, accessToken }: FeedbackPanelProps) {
  const api = useTracesApi(accessToken);
  const traceRef = summary.trace_ref ?? "";
  const feedback = useQuery({
    queryKey: ["traceFeedbackDetail", accessToken, summary.trace_id, traceRef],
    queryFn: () => api.feedback(summary.trace_id, traceRef),
    retry: false,
  });
  const view = feedback.data ? feedbackView(feedback.data.feedback) : null;
  if (!view) return null;
  const low = isLow(view.lowest);
  return (
    <section
      aria-label="User feedback"
      data-low={low || undefined}
      className={cn(
        "flex shrink-0 flex-col gap-2 border-b px-4 py-3",
        low ? "bg-destructive/[0.04] shadow-[inset_2px_0_0_var(--color-destructive)]" : "bg-muted/30",
      )}
    >
      <header className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
        <MessageSquareQuote className="size-3.5" />
        User feedback
        {view.entries.length > 1 && (
          <span className={cn("inline-flex items-center gap-1 font-mono", low && "text-destructive")}>
            <Star className="size-3" />
            {formatScore(view.average)}/10 avg from {view.entries.length} users
          </span>
        )}
      </header>
      <ul className="flex max-h-48 flex-col divide-y overflow-y-auto">
        {view.entries.map((entry) => (
          <Entry key={entry.author} entry={entry} />
        ))}
      </ul>
    </section>
  );
}
