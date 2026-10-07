"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { apiClient } from "@/components/networking";
import { Button } from "@/components/ui/button";
import type { components, paths } from "@/lib/http/schema";

import { useTracesLive } from "../../api";
import { Section } from "./Section";

type Summary = components["schemas"]["FeedbackSummary"];
type Feedback = components["schemas"]["Feedback"];
type Create = components["schemas"]["FeedbackCreate"];
type Target = Required<NonNullable<paths["/v1/feedback/summary"]["get"]["parameters"]["query"]>>;

function RatingForm({
  accessToken,
  target,
  summary,
  onSaved,
}: {
  accessToken: string;
  target: Target;
  summary: Summary;
  onSaved: () => Promise<void>;
}) {
  const [value, setValue] = useState<number | null>(summary.mine?.value ?? null);
  const [comment, setComment] = useState(summary.mine?.comment ?? "");
  const save = useMutation({
    mutationFn: () => {
      if (value === null) throw new Error("Choose a score from 0 to 10");
      return apiClient.post<Feedback>("/v1/feedback", {
        accessToken,
        body: { ...target, key: "usefulness", value, comment } satisfies Create,
      });
    },
    onSuccess: onSaved,
  });
  const remove = useMutation({
    mutationFn: () => apiClient.delete(`/v1/feedback/${encodeURIComponent(summary.mine!.id)}`, { accessToken }),
    onSuccess: onSaved,
  });
  const busy = save.isPending || remove.isPending;
  const actionLabel = summary.mine ? "Update rating" : "Save rating";
  return (
    <form
      className="flex flex-col gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        save.mutate();
      }}
    >
      <fieldset disabled={busy} className="flex flex-col gap-2">
        <legend className="mb-2 text-sm">How useful was this response?</legend>
        <div className="flex flex-wrap gap-1.5">
          {Array.from({ length: 11 }, (_, score) => (
            <label key={score} className="cursor-pointer">
              <input
                className="peer sr-only"
                type="radio"
                name="usefulness"
                value={score}
                checked={value === score}
                onChange={() => setValue(score)}
                aria-label={`${score} out of 10`}
              />
              <span className="flex size-8 items-center justify-center rounded-md border text-sm peer-checked:border-primary peer-checked:bg-primary peer-checked:text-primary-foreground peer-focus-visible:ring-2 peer-focus-visible:ring-ring">
                {score}
              </span>
            </label>
          ))}
        </div>
        <p className="text-xs text-muted-foreground">0 = Not useful · 10 = Extremely useful</p>
        <label className="flex flex-col gap-1 text-sm">
          Comment (optional)
          <textarea
            className="min-h-16 rounded-md border bg-background p-2"
            maxLength={4000}
            value={comment}
            onChange={(event) => setComment(event.target.value)}
          />
        </label>
      </fieldset>
      <div className="flex gap-2">
        <Button type="submit" size="sm" disabled={value === null || busy}>
          {save.isPending ? "Saving…" : actionLabel}
        </Button>
        {summary.mine && (
          <Button type="button" variant="outline" size="sm" disabled={busy} onClick={() => remove.mutate()}>
            Remove rating
          </Button>
        )}
      </div>
      {(save.error || remove.error) && <p role="alert">{(save.error || remove.error)?.message}</p>}
    </form>
  );
}

export function UsefulnessFeedback({
  accessToken,
  traceId,
  traceRef,
  spanId,
}: {
  accessToken: string;
  traceId: string;
  traceRef?: string;
  spanId: string;
}) {
  const live = useTracesLive();
  const client = useQueryClient();
  const target = { trace_id: traceId, trace_ref: traceRef ?? "", span_id: spanId } satisfies Target;
  const queryKey = ["lensFeedback", accessToken, traceId, traceRef, spanId];
  const queryOptions = {
    queryKey,
    queryFn: () => apiClient.get<Summary>("/v1/feedback/summary", { accessToken, query: target }),
    enabled: live,
    retry: false,
  };
  const query = useQuery(queryOptions);
  // Fixed demo snapshots must never send feedback to a live project.
  if (!live) return null;
  return (
    <Section title="Usefulness">
      {query.isPending && <p role="status">Loading ratings…</p>}
      {query.isError && (
        <div role="alert">
          Could not load ratings: {query.error.message}
          <Button variant="outline" size="sm" onClick={() => void query.refetch()}>
            Retry
          </Button>
        </div>
      )}
      {query.data && (
        <>
          <p role="status" className="text-sm">
            {query.data.average === null
              ? "Unrated"
              : `${query.data.average.toFixed(1)} / 10 · ${query.data.count} ${query.data.count === 1 ? "rating" : "ratings"}`}
          </p>
          {query.data.count > 0 && (
            <details className="text-xs text-muted-foreground">
              <summary className="cursor-pointer">Score distribution</summary>
              <ul className="mt-2 flex flex-wrap gap-3" aria-label="Score distribution">
                {Object.entries(query.data.distribution).map(([score, count]) => (
                  <li key={score}>
                    {score}: {count}
                  </li>
                ))}
              </ul>
            </details>
          )}
          {query.data.can_rate ? (
            <RatingForm
              key={`${traceId}:${traceRef}:${spanId}:${query.data.mine?.updated_at ?? "unrated"}`}
              accessToken={accessToken}
              target={target}
              summary={query.data}
              onSaved={() => client.invalidateQueries({ queryKey })}
            />
          ) : (
            <p className="text-xs text-muted-foreground">
              Read-only access. You can view ratings but cannot change them.
            </p>
          )}
        </>
      )}
    </Section>
  );
}
