import { describe, expect, it } from "vitest";

import { LOW_SCORE, isLowFeedback, type TraceFeedbackState } from "./useTraceFeedback";

const ready = (count: number, lowest: number | null): TraceFeedbackState => ({
  status: "ready",
  summary: { count, average: lowest, lowest },
});

describe("isLowFeedback", () => {
  it.each([
    { name: "a run an end user scored at the threshold", state: ready(1, LOW_SCORE), low: true },
    { name: "a run whose lowest score is just above the threshold", state: ready(3, LOW_SCORE + 1), low: false },
    { name: "a run nobody scored", state: ready(0, null), low: false },
    { name: "a run whose feedback is still loading", state: { status: "pending" } as const, low: false },
    { name: "a run whose feedback failed to load", state: { status: "error" } as const, low: false },
    { name: "a run with no feedback state", state: undefined, low: false },
  ])("flags $name only when it is low", ({ state, low }) => {
    expect(isLowFeedback(state)).toBe(low);
  });
});
