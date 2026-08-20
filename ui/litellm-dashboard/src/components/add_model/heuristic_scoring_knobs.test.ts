import { describe, expect, it } from "vitest";

import { heuristicScoringRoleFor } from "./ComplexityRouterConfig";
import {
  dimensionLabel,
  hydrateDimensionWeights,
  hydrateReasoningOverrideMinScore,
  hydrateTierBoundaries,
  hydrateTokenThresholds,
  weightTotal,
} from "./heuristic_scoring_knobs";

describe("hydrating the scorer knobs", () => {
  // The tri-state rests on this: hydrating an absent knob to the shipped defaults would make an untouched
  // save write them out and pin the router to whatever they were when the modal was opened.
  it.each([[undefined], [null], ["0.15"], [[0.15]]])("hydrates %s to undefined, not to the defaults", (raw) => {
    expect(hydrateTierBoundaries(raw)).toBeUndefined();
    expect(hydrateDimensionWeights(raw)).toBeUndefined();
  });

  it("keeps a stored dict exactly as stored, including negatives and zero", () => {
    expect(hydrateTierBoundaries({ simple_medium: -1, medium_complex: 0, complex_reasoning: 0.6 })).toEqual({
      simple_medium: -1,
      medium_complex: 0,
      complex_reasoning: 0.6,
    });
  });

  it("leaves a partial dict partial, since the backend fills the rest at scoring time", () => {
    expect(hydrateTokenThresholds({ complex: 900 })).toEqual({ complex: 900 });
  });

  it("drops non-numeric and non-finite entries", () => {
    expect(hydrateTokenThresholds({ simple: "25", complex: Number.NaN, other: 900 })).toEqual({ other: 900 });
  });

  it("preserves a key it does not recognise rather than deleting an operator's config", () => {
    // The dimension set is the proxy's, not the dashboard's, so an unknown key may be a newer backend
    // rather than a typo. It is kept, and simply has no control rendered for it.
    expect(hydrateDimensionWeights({ codePresence: 0.3, somethingNew: 0.4 })).toEqual({
      codePresence: 0.3,
      somethingNew: 0.4,
    });
  });

  it("totals weights and rounds away float drift", () => {
    expect(weightTotal({ a: 0.1, b: 0.2 })).toBe(0.3);
  });

  it.each([[undefined], [null], ["0.15"], [Number.NaN], [Number.POSITIVE_INFINITY], [{ value: 0.15 }]])(
    "hydrates the reasoning override floor %s to undefined",
    (raw) => {
      expect(hydrateReasoningOverrideMinScore(raw)).toBeUndefined();
    },
  );

  // A stored 0 is an unconditional override, so hydrating it to undefined would silently retune the router
  // back to tracking simple_medium on the next save.
  it("hydrates a stored reasoning override floor, zero and negatives included", () => {
    expect(hydrateReasoningOverrideMinScore(0)).toBe(0);
    expect(hydrateReasoningOverrideMinScore(-0.3)).toBe(-0.3);
    expect(hydrateReasoningOverrideMinScore(0.42)).toBe(0.42);
  });

  it("falls back to the raw key when a dimension has no label yet", () => {
    expect(dimensionLabel("codePresence")).toBe("Code presence");
    expect(dimensionLabel("somethingNew")).toBe("somethingNew");
  });
});

describe("heuristicScoringRoleFor", () => {
  it.each([
    ["heuristic", undefined, "decides"],
    ["heuristic", "default_model", "decides"],
    ["llm", undefined, "fallback_only"],
    ["llm", "heuristic", "fallback_only"],
    ["llm", "default_model", "never"],
  ] as const)("classifier %s with fallback %s scores as %s", (type, fallback, expected) => {
    expect(heuristicScoringRoleFor(type, fallback)).toBe(expected);
  });
});
