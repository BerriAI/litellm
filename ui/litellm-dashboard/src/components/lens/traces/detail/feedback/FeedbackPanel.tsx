"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { MessageSquareText } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/cva.config";
import { toast } from "@/lib/toast";
import { formatActivityTimestamp } from "@/utils/activityTimestamp";

import { useTracesApi } from "../../api";
import { LOW_SCORE, traceFeedbackKey } from "../../list/useTraceFeedback";
import type { Feedback, TraceSummary } from "../../types";
import { SCORES, feedbackView } from "./feedback";

interface FeedbackPanelProps {
  summary: Pick<TraceSummary, "trace_id" | "trace_ref">;
  accessToken: string;
}

const scoreTone = (score: number) => (score <= LOW_SCORE ? "text-destructive" : "text-foreground");

function pickerTone(score: number, selected: boolean): string {
  if (!selected) return "hover:bg-muted";
  if (score <= LOW_SCORE) return "bg-destructive text-white ring-destructive";
  return "bg-foreground text-background ring-foreground";
}

function Entry({ entry, label }: { entry: Feedback; label: string }) {
  return (
    <li className="flex flex-col gap-1 border-b py-2 last:border-b-0" data-testid="feedback-entry">
      <div className="flex items-baseline justify-between gap-2 text-xs">
        <span className="truncate font-medium">{label}</span>
        <span className={cn("font-mono tabular-nums", scoreTone(entry.score))}>{entry.score}/10</span>
      </div>
      {entry.comment && <p className="text-xs whitespace-pre-wrap text-muted-foreground">{entry.comment}</p>}
      <span className="text-xs text-muted-foreground/70" title={entry.updated_at}>
        {formatActivityTimestamp(entry.updated_at)}
      </span>
    </li>
  );
}

function ScorePicker({ value, onChange }: { value: number | null; onChange: (score: number) => void }) {
  return (
    <div role="radiogroup" aria-label="Score from 0 to 10" className="grid grid-cols-11 gap-0.5">
      {SCORES.map((score) => (
        <button
          key={score}
          type="button"
          role="radio"
          aria-checked={value === score}
          onClick={() => onChange(score)}
          className={cn(
            "h-7 rounded text-xs tabular-nums ring-1 ring-border transition-colors",
            pickerTone(score, value === score),
          )}
        >
          {score}
        </button>
      ))}
    </div>
  );
}

function FeedbackForm({
  mine,
  saving,
  removing,
  onSave,
  onRemove,
}: {
  mine: Feedback | null;
  saving: boolean;
  removing: boolean;
  onSave: (score: number, comment: string) => void;
  onRemove: () => void;
}) {
  const [score, setScore] = useState<number | null>(mine?.score ?? null);
  const [comment, setComment] = useState(mine?.comment ?? "");
  return (
    <form
      className="flex flex-col gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        if (score !== null) onSave(score, comment);
      }}
    >
      <span className="text-xs font-medium">{mine ? "Your feedback" : "Rate this run"}</span>
      <ScorePicker value={score} onChange={setScore} />
      <Textarea
        aria-label="Feedback comment"
        placeholder="What went well or wrong?"
        value={comment}
        onChange={(event) => setComment(event.target.value)}
        className="min-h-16 text-xs"
      />
      <div className="flex justify-end gap-1.5">
        {mine && (
          <Button type="button" variant="ghost" size="xs" disabled={removing} onClick={onRemove}>
            Remove
          </Button>
        )}
        <Button type="submit" size="xs" disabled={score === null || saving}>
          {mine ? "Update" : "Save"}
        </Button>
      </div>
    </form>
  );
}

/** Human 0–10 scores and comments on one run, plus the caller's own editable entry. */
export function FeedbackPanel({ summary, accessToken }: FeedbackPanelProps) {
  const api = useTracesApi(accessToken);
  const queryClient = useQueryClient();
  const traceRef = summary.trace_ref ?? "";
  const queryKey = ["traceFeedbackDetail", accessToken, summary.trace_id, traceRef];
  const feedback = useQuery({ queryKey, queryFn: () => api.feedback(summary.trace_id, traceRef), retry: false });
  const refresh = () =>
    Promise.all([
      queryClient.invalidateQueries({ queryKey }),
      queryClient.invalidateQueries({ queryKey: traceFeedbackKey(accessToken) }),
    ]);
  const save = useMutation({
    mutationFn: ({ score, comment }: { score: number; comment: string }) =>
      api.submitFeedback({ trace_id: summary.trace_id, trace_ref: traceRef, score, comment }),
    onSuccess: () => toast.success("Feedback saved"),
    onError: (error) => toast.fromError(error),
    onSettled: refresh,
  });
  const remove = useMutation({
    mutationFn: () => api.deleteFeedback(summary.trace_id, traceRef),
    onSuccess: () => toast.success("Feedback removed"),
    onError: (error) => toast.fromError(error),
    onSettled: refresh,
  });
  const view = feedback.data ? feedbackView(feedback.data) : null;
  const count = feedback.data?.feedback.length ?? 0;
  const low = feedback.data?.feedback.some((entry) => entry.score <= LOW_SCORE) ?? false;
  return (
    <Popover>
      <PopoverTrigger
        render={
          <Button
            variant="outline"
            size="xs"
            className={cn("h-7 shrink-0 gap-1.5 text-xs shadow-none", low && "border-destructive/40 text-destructive")}
          />
        }
      >
        <MessageSquareText className="size-3" />
        Feedback
        {count > 0 && (
          <span className="rounded bg-muted px-1 font-mono tabular-nums" data-testid="feedback-count">
            {view?.average?.toFixed(1)} · {count}
          </span>
        )}
      </PopoverTrigger>
      <PopoverContent align="end" className="w-80 gap-3" aria-label="Run feedback">
        {feedback.isPending && <p className="text-xs text-muted-foreground">Loading feedback…</p>}
        {feedback.isError && (
          <p role="alert" className="text-xs text-destructive">
            Could not load feedback: {feedback.error.message}
          </p>
        )}
        {view && (
          <>
            {view.others.length > 0 ? (
              <ul aria-label="Feedback from others" className="max-h-64 overflow-y-auto">
                {view.others.map((entry) => (
                  <Entry key={entry.author} entry={entry} label={entry.author} />
                ))}
              </ul>
            ) : (
              !view.mine && <p className="text-xs text-muted-foreground">No feedback on this run yet.</p>
            )}
            <FeedbackForm
              key={view.mine?.updated_at ?? "new"}
              mine={view.mine}
              saving={save.isPending}
              removing={remove.isPending}
              onSave={(score, comment) => save.mutate({ score, comment })}
              onRemove={() => remove.mutate()}
            />
          </>
        )}
      </PopoverContent>
    </Popover>
  );
}
