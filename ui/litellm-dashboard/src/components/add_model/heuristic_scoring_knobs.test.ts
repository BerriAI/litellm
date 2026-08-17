import { describe, expect, it } from "vitest";

import {
  DEFAULT_DIMENSION_WEIGHTS,
  DEFAULT_TIER_BOUNDARIES,
  DEFAULT_TOKEN_THRESHOLDS,
  heuristicScoringRoleFor,
} from "./ComplexityRouterConfig";
import {
  hydrateDimensionWeights,
  hydrateTierBoundaries,
  hydrateTokenThresholds,
  weightTotal,
} from "./heuristic_scoring_knobs";

describe("hydrating the heuristic scorer knobs", () => {
  // The whole tri-state rests on this: an absent knob must stay absent, because hydrating the defaults
  // would make an untouched save write them out and pin the router to today's numbers forever.
  it.each([
    ["absent", undefined],
    ["null", null],
    ["a string", "0.15"],
    ["an array", [0.15, 0.35]],
  ])("hydrates %s to undefined rather than to the defaults", (_label, raw) => {
    expect(hydrateTierBoundaries(raw)).toBeUndefined();
    expect(hydrateTokenThresholds(raw)).toBeUndefined();
    expect(hydrateDimensionWeights(raw)).toBeUndefined();
  });

  it("round-trips explicit stored boundaries unchanged", () => {
    expect(hydrateTierBoundaries({ simple_medium: 0.22, medium_complex: 0.44, complex_reasoning: 0.66 })).toEqual({
      simple_medium: 0.22,
      medium_complex: 0.44,
      complex_reasoning: 0.66,
    });
  });

  it("keeps a negative boundary, which is how trivial prompts get lifted a tier", () => {
    expect(hydrateTierBoundaries({ simple_medium: -1, medium_complex: -0.9, complex_reasoning: -0.8 })).toEqual({
      simple_medium: -1,
      medium_complex: -0.9,
      complex_reasoning: -0.8,
    });
  });

  it("keeps a zero weight, which switches a dimension off", () => {
    expect(hydrateDimensionWeights({ ...DEFAULT_DIMENSION_WEIGHTS, codePresence: 0 })?.codePresence).toBe(0);
  });

  it("fills only the missing keys of a partial dict, matching the backend's own defaulted reads", () => {
    expect(hydrateTierBoundaries({ medium_complex: 0.44 })).toEqual({
      simple_medium: DEFAULT_TIER_BOUNDARIES.simple_medium,
      medium_complex: 0.44,
      complex_reasoning: DEFAULT_TIER_BOUNDARIES.complex_reasoning,
    });
    expect(hydrateTokenThresholds({ complex: 900 })).toEqual({
      simple: DEFAULT_TOKEN_THRESHOLDS.simple,
      complex: 900,
    });
  });

  it("drops a non-numeric or non-finite entry back to its default", () => {
    expect(hydrateTokenThresholds({ simple: "25", complex: Number.NaN })).toEqual(DEFAULT_TOKEN_THRESHOLDS);
  });

  it("ignores a key the scorer does not read, so a typo cannot reach the payload", () => {
    // weights.get(name, 0) in the backend silently gives an unknown dimension weight 0, so a misspelled
    // key must never be carried through as if it were configuration.
    expect(hydrateDimensionWeights({ ...DEFAULT_DIMENSION_WEIGHTS, codePresense: 0.9 })).toEqual(
      DEFAULT_DIMENSION_WEIGHTS,
    );
  });
});

describe("weightTotal", () => {
  it("totals the shipped defaults to exactly 1", () => {
    expect(weightTotal(DEFAULT_DIMENSION_WEIGHTS)).toBe(1);
  });

  it("rounds away float drift so the readout cannot show 1.0000000000000002", () => {
    expect(weightTotal({ ...DEFAULT_DIMENSION_WEIGHTS, codePresence: 0.5 })).toBe(1.2);
  });
});

describe("heuristicScoringRoleFor", () => {
  it.each([
    ["heuristic", undefined, "decides"],
    ["heuristic", "default_model", "decides"],
    ["heuristic", "heuristic", "decides"],
    ["llm", "heuristic", "fallback_only"],
    ["llm", undefined, "fallback_only"],
    ["llm", "default_model", "never"],
  ] as const)("classifier %s with fallback %s scores as %s", (classifierType, fallback, expected) => {
    expect(heuristicScoringRoleFor(classifierType, fallback)).toBe(expected);
  });
});
