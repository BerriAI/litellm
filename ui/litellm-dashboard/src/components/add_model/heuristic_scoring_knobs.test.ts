import { describe, expect, it } from "vitest";

import { heuristicScoringRoleFor } from "./ComplexityRouterConfig";
import {
  dimensionLabel,
  effectiveDimensionWeights,
  rebalanceDimensionWeights,
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

describe("rebalancing the complete weight vector", () => {
  const defaults = {
    codePresence: 0.3,
    reasoningMarkers: 0.25,
    technicalTerms: 0.25,
    tokenCount: 0.1,
    simpleIndicators: 0.05,
    multiStepPatterns: 0.03,
    questionComplexity: 0.02,
  };
  const row = { id: "domain", name: "domainMarkers", weight: 0.2, keywords: ["orbitmesh"] };
  const success = (result: ReturnType<typeof rebalanceDimensionWeights>) => {
    if (!result.ok) throw new Error(result.error);
    const total =
      Object.keys(defaults).reduce((sum, name) => sum + result.dimension_weights[name], 0) +
      (result.custom_dimensions ?? []).reduce((sum, dimension) => sum + dimension.weight, 0);
    expect(total).toBeCloseTo(1, 12);
    return result;
  };

  it("adds a dimension, edits either kind, and redistributes its share on removal", () => {
    const added = success(rebalanceDimensionWeights(defaults, undefined, undefined, { type: "add", row }));
    expect(added.dimension_weights.codePresence).toBeCloseTo(0.24, 12);
    expect(added.custom_dimensions?.[0].weight).toBe(0.2);
    const edited = success(
      rebalanceDimensionWeights(defaults, added.dimension_weights, added.custom_dimensions, {
        type: "set",
        target: { kind: "custom", id: row.id },
        weight: 0.4,
      }),
    );
    expect(edited.dimension_weights.codePresence).toBeCloseTo(0.18, 12);
    const builtin = success(
      rebalanceDimensionWeights(defaults, edited.dimension_weights, edited.custom_dimensions, {
        type: "set",
        target: { kind: "builtin", id: "codePresence" },
        weight: 0.5,
      }),
    );
    expect(builtin.dimension_weights.codePresence).toBe(0.5);
    expect(builtin.custom_dimensions?.[0].weight).toBeCloseTo((0.4 * 0.5) / 0.82, 12);
    const removed = success(
      rebalanceDimensionWeights(defaults, added.dimension_weights, added.custom_dimensions, {
        type: "remove",
        id: row.id,
      }),
    );
    expect(removed.custom_dimensions).toBeUndefined();
    expect(removed.dimension_weights.codePresence).toBeCloseTo(defaults.codePresence, 12);
  });

  it("uses zero for omitted keys of an explicit map and preserves ignored keys", () => {
    expect(effectiveDimensionWeights(defaults, { codePresence: 0.2 }).technicalTerms).toBe(0);
    expect(effectiveDimensionWeights(defaults, undefined)).toEqual(defaults);
    const result = success(
      rebalanceDimensionWeights(defaults, { codePresence: 0.2, unknown: 7 }, undefined, { type: "add", row }),
    );
    expect(result.dimension_weights.unknown).toBe(7);
    expect(result.dimension_weights.codePresence).toBeCloseTo(0.8, 12);
    expect(result.dimension_weights.technicalTerms).toBe(0);
  });

  it("distributes an all-zero vector without inventing custom matchers", () => {
    const result = success(rebalanceDimensionWeights(defaults, {}, undefined, { type: "add", row }));
    expect(result.dimension_weights.codePresence).toBeCloseTo(0.8 / 7, 12);
    expect(result.custom_dimensions).toEqual([row]);
  });

  it("keeps small positive custom weights and full precision", () => {
    const tiny = { ...row, weight: 1e-8 };
    const result = success(
      rebalanceDimensionWeights(defaults, defaults, [tiny], {
        type: "set",
        target: { kind: "builtin", id: "codePresence" },
        weight: 0.123456789,
      }),
    );
    expect(result.dimension_weights.codePresence).toBe(0.123456789);
    expect(result.custom_dimensions?.[0].weight).toBeGreaterThan(0);
    expect(result.custom_dimensions?.[0].weight).toBeLessThan(0.01);
  });

  it.each([0, 1, Number.NaN, -0.1, 1.1])(
    "rejects invalid or impossible custom weight %s without changing the input",
    (weight) => {
      const original = structuredClone(row);
      const result = rebalanceDimensionWeights(defaults, defaults, [row, { ...row, id: "other" }], {
        type: "set",
        target: { kind: "custom", id: row.id },
        weight,
      });
      expect(result.ok).toBe(false);
      expect(row).toEqual(original);
    },
  );

  it("allows one dimension to take the whole budget when no custom sibling needs a share", () => {
    const result = success(
      rebalanceDimensionWeights(defaults, defaults, [row], {
        type: "set",
        target: { kind: "custom", id: row.id },
        weight: 1,
      }),
    );
    expect(Object.values(result.dimension_weights).every((weight) => weight === 0)).toBe(true);
    expect(result.custom_dimensions?.[0].weight).toBe(1);
  });

  it("refuses missing defaults and invalid stored weights, preserving absence on built-in edits", () => {
    const edit = { type: "set", target: { kind: "builtin", id: "codePresence" }, weight: 0.2 } as const;
    expect(rebalanceDimensionWeights(undefined, defaults, undefined, edit).ok).toBe(false);
    expect(rebalanceDimensionWeights(defaults, { codePresence: -1 }, undefined, edit).ok).toBe(false);
    expect(success(rebalanceDimensionWeights(defaults, undefined, undefined, edit)).custom_dimensions).toBeUndefined();
    expect(success(rebalanceDimensionWeights(defaults, undefined, [], edit)).custom_dimensions).toEqual([]);
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
