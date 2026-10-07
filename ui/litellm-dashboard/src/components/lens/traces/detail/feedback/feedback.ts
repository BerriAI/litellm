import type { Feedback } from "../../types";

export interface FeedbackView {
  readonly entries: readonly Feedback[];
  readonly average: number;
  readonly lowest: number;
}

/** End-user feedback on one run, newest first, or null when nobody has rated it. */
export function feedbackView(feedback: readonly Feedback[]): FeedbackView | null {
  if (feedback.length === 0) return null;
  const scores = feedback.map((entry) => entry.score);
  return {
    entries: [...feedback].sort((a, b) => Date.parse(b.updated_at) - Date.parse(a.updated_at)),
    average: scores.reduce((sum, score) => sum + score, 0) / scores.length,
    lowest: Math.min(...scores),
  };
}
