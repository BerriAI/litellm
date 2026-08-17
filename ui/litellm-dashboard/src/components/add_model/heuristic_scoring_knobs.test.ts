import { describe, expect, it } from "vitest";

import { DEFAULT_DIMENSION_WEIGHTS, DEFAULT_TIER_BOUNDARIES, heuristicScoringRoleFor } from "./ComplexityRouterConfig";
import {
  hydrateDimensionWeights,
  hydrateTierBoundaries,
  hydrateTokenThresholds,
  weightTotal,
} from "./heuristic_scoring_knobs";

describe("hydrating the scorer knobs", () => {
  // The tri-state rests on this: hydrating an absent knob to the defaults would make an untouched save
  // write them out and pin the router to today's numbers forever.
  it.each([[undefined], [null], ["0.15"], [[0.15]]])("hydrates %s to undefined, not to the defaults", (raw) => {
    expect(hydrateTierBoundaries(raw)).toBeUndefined();
    expect(hydrateDimensionWeights(raw)).toBeUndefined();
  });

  it("round-trips explicit values, including negatives and zero", () => {
    expect(hydrateTierBoundaries({ simple_medium: -1, medium_complex: -0.9, complex_reasoning: 0 })).toEqual({
      simple_medium: -1,
      medium_complex: -0.9,
      complex_reasoning: 0,
    });
  });

  it("fills only missing keys, matching the backend's own defaulted reads, and drops junk", () => {
    expect(hydrateTokenThresholds({ complex: 900, simple: "25" })).toEqual({ simple: 15, complex: 900 });
    expect(hydrateTierBoundaries({ medium_complex: 0.44 })).toEqual({
      ...DEFAULT_TIER_BOUNDARIES,
      medium_complex: 0.44,
    });
  });

  it("ignores an unknown key, so a typo cannot reach the payload", () => {
    // weights.get(name, 0) on the backend silently gives an unknown dimension weight 0.
    expect(hydrateDimensionWeights({ ...DEFAULT_DIMENSION_WEIGHTS, codePresense: 0.9 })).toEqual(
      DEFAULT_DIMENSION_WEIGHTS,
    );
  });

  it("totals the shipped weights to 1 and rounds away float drift", () => {
    expect(weightTotal(DEFAULT_DIMENSION_WEIGHTS)).toBe(1);
    expect(weightTotal({ ...DEFAULT_DIMENSION_WEIGHTS, codePresence: 0.5 })).toBe(1.2);
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
