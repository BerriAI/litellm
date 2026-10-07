import { describe, expect, it } from "vitest";

import { LOW_SCORE, matchesFeedback, type TraceFeedbackState } from "./useTraceFeedback";

const ready = (count: number, lowest: number | null): TraceFeedbackState => ({
  status: "ready",
  summary: { count, average: lowest, lowest },
});

describe("matchesFeedback", () => {
  it.each([
    { name: "unrated", state: ready(0, null), rated: false, low: false },
    { name: "rated above the threshold", state: ready(2, LOW_SCORE + 1), rated: true, low: false },
    { name: "rated at the threshold", state: ready(1, LOW_SCORE), rated: true, low: true },
    { name: "still loading", state: { status: "pending" } as const, rated: false, low: false },
    { name: "failed to load", state: { status: "error" } as const, rated: false, low: false },
  ])("keeps a $name run only under the filters that should show it", ({ state, rated, low }) => {
    expect([matchesFeedback(state, "all"), matchesFeedback(state, "rated"), matchesFeedback(state, "low")]).toEqual([
      true,
      rated,
      low,
    ]);
  });
});
