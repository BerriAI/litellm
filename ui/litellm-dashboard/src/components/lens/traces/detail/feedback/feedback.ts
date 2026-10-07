import type { Feedback, TraceFeedback } from "../../types";

export const SCORES = Array.from({ length: 11 }, (_, score) => score);

export interface FeedbackView {
  readonly mine: Feedback | null;
  readonly others: readonly Feedback[];
  readonly average: number | null;
}

/** Splits the caller's own entry from everyone else's, newest first, and averages every score. */
export function feedbackView({ feedback, viewer }: Pick<TraceFeedback, "feedback" | "viewer">): FeedbackView {
  const newestFirst = [...feedback].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at));
  const mine = newestFirst.find((entry) => viewer !== "" && entry.author === viewer) ?? null;
  const total = feedback.reduce((sum, entry) => sum + entry.score, 0);
  return {
    mine,
    others: newestFirst.filter((entry) => entry !== mine),
    average: feedback.length ? total / feedback.length : null,
  };
}
