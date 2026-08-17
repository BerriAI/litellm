import { describe, expect, it } from "vitest";

import { normalizeTierModels, resolveComplexityDefaultModel } from "./complexity_router_tiers";

import type { ComplexityTiers } from "./ComplexityRouterConfig";

// The backend types a tier as `str | list[str]` and widens with
// `models if isinstance(models, list) else [models]`
// (litellm/router_strategy/complexity_router/config.py:255, :441). These cases assert the
// expected verdict per input rather than just agreement between call sites, so the test still
// has teeth if every reader were changed at once.
describe("normalizeTierModels", () => {
  it("widens a pinned single model to a one-element pool", () => {
    expect(normalizeTierModels("gpt-4o-mini")).toEqual(["gpt-4o-mini"]);
  });

  it("passes a pool through in order", () => {
    expect(normalizeTierModels(["a", "b"])).toEqual(["a", "b"]);
  });

  it("treats an empty string as no models, not a pool containing an empty name", () => {
    expect(normalizeTierModels("")).toEqual([]);
  });

  it("drops non-string entries rather than typing them as models", () => {
    expect(normalizeTierModels(["a", 3, null, "b"])).toEqual(["a", "b"]);
  });

  it.each([[undefined], [null], [{}], [42]])("returns no models for %s", (value) => {
    expect(normalizeTierModels(value)).toEqual([]);
  });
});

// router.py derives the default as `MEDIUM or SIMPLE` and raises when neither holds a model, so
// the resolver must not invent a COMPLEX/REASONING fallthrough the backend would never take.
describe("resolveComplexityDefaultModel", () => {
  const tiers: ComplexityTiers = {
    SIMPLE: ["simple-model"],
    MEDIUM: ["medium-model"],
    COMPLEX: ["complex-model"],
    REASONING: ["reasoning-model"],
  };
  const noTiers: ComplexityTiers = { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] };

  it("derives from MEDIUM first when nothing is pinned", () => {
    expect(resolveComplexityDefaultModel(tiers)).toBe("medium-model");
  });

  it("falls back to SIMPLE when MEDIUM is empty", () => {
    expect(resolveComplexityDefaultModel({ ...tiers, MEDIUM: [] })).toBe("simple-model");
  });

  it("derives nothing from COMPLEX or REASONING, which the backend never falls through to", () => {
    expect(resolveComplexityDefaultModel({ ...tiers, MEDIUM: [], SIMPLE: [] })).toBeUndefined();
  });

  it("lets a pin beat the tiers rather than merely filling in for them", () => {
    expect(resolveComplexityDefaultModel(tiers, "pinned-model")).toBe("pinned-model");
  });

  it("stands alone as the default when no tier holds a model", () => {
    expect(resolveComplexityDefaultModel(noTiers, "pinned-model")).toBe("pinned-model");
  });

  it.each([[""], ["   "], [undefined]])("reads %o as no pin and goes back to the tiers", (pinned) => {
    expect(resolveComplexityDefaultModel(tiers, pinned)).toBe("medium-model");
  });

  it("resolves to nothing when neither a pin nor a tier offers a model", () => {
    expect(resolveComplexityDefaultModel(noTiers)).toBeUndefined();
  });
});
