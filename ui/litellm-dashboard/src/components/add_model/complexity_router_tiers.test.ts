import { describe, expect, it } from "vitest";

import {
  hydrateTierModelParams,
  normalizeTierModels,
  pruneTierModelParams,
  serializeTierModelConfigs,
  setTierModelReasoningEffort,
} from "./complexity_router_tiers";
import { resolveComplexityDefaultModel } from "./tier_rows";

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
    expect(resolveComplexityDefaultModel({ tiers: tiers })).toBe("medium-model");
  });

  it("falls back to SIMPLE when MEDIUM is empty", () => {
    expect(resolveComplexityDefaultModel({ tiers: { ...tiers, MEDIUM: [] } })).toBe("simple-model");
  });

  it("derives nothing from COMPLEX or REASONING, which the backend never falls through to", () => {
    expect(resolveComplexityDefaultModel({ tiers: { ...tiers, MEDIUM: [], SIMPLE: [] } })).toBeUndefined();
  });

  it("lets a pin beat the tiers rather than merely filling in for them", () => {
    expect(resolveComplexityDefaultModel({ tiers: tiers }, "pinned-model")).toBe("pinned-model");
  });

  it("stands alone as the default when no tier holds a model", () => {
    expect(resolveComplexityDefaultModel({ tiers: noTiers }, "pinned-model")).toBe("pinned-model");
  });

  it.each([[""], ["   "], [undefined]])("reads %o as no pin and goes back to the tiers", (pinned) => {
    expect(resolveComplexityDefaultModel({ tiers: tiers }, pinned)).toBe("medium-model");
  });

  it("resolves to nothing when neither a pin nor a tier offers a model", () => {
    expect(resolveComplexityDefaultModel({ tiers: noTiers })).toBeUndefined();
  });
});

// The backend also accepts `{model_name, litellm_params}` entries and splits them into the
// sibling tier_model_configs key at validation (config.py `_normalize_tier_model_configs`).
// Before this widening, an object entry was silently dropped here, so opening the edit modal on
// a yaml-authored config rendered the tier empty and the next save destroyed it.
describe("normalizeTierModels object entries", () => {
  it("reads model_name from an object entry the way the backend does", () => {
    expect(normalizeTierModels([{ model_name: "opus", litellm_params: { reasoning_effort: "high" } }, "mini"])).toEqual(
      ["opus", "mini"],
    );
  });

  it("widens a single object entry to a one-element pool", () => {
    expect(normalizeTierModels({ model_name: "opus" })).toEqual(["opus"]);
  });

  it("drops an object without a model_name", () => {
    expect(normalizeTierModels([{ litellm_params: { reasoning_effort: "high" } }])).toEqual([]);
  });
});

describe("hydrateTierModelParams", () => {
  it("reads the sibling tier_model_configs key", () => {
    expect(
      hydrateTierModelParams(
        { MEDIUM: ["opus"] },
        { MEDIUM: [{ model_name: "opus", litellm_params: { reasoning_effort: "medium" } }] },
      ),
    ).toEqual({ MEDIUM: { opus: { reasoning_effort: "medium" } } });
  });

  it("reads inline object entries out of tiers", () => {
    expect(
      hydrateTierModelParams(
        { COMPLEX: [{ model_name: "opus", litellm_params: { reasoning_effort: "high" } }] },
        undefined,
      ),
    ).toEqual({ COMPLEX: { opus: { reasoning_effort: "high" } } });
  });

  // config.py merges the two sources with tier_model_configs winning per (tier, model); hydrating
  // the other way round would show the operator a value the router never uses.
  it("lets tier_model_configs beat an inline entry for the same tier and model", () => {
    expect(
      hydrateTierModelParams(
        { MEDIUM: [{ model_name: "opus", litellm_params: { reasoning_effort: "low" } }] },
        { MEDIUM: [{ model_name: "opus", litellm_params: { reasoning_effort: "medium" } }] },
      ),
    ).toEqual({ MEDIUM: { opus: { reasoning_effort: "medium" } } });
  });

  it("hydrates to undefined when nothing carries params, so an untouched save stays byte-identical", () => {
    expect(
      hydrateTierModelParams({ SIMPLE: ["mini"], MEDIUM: [{ model_name: "opus", litellm_params: {} }] }, undefined),
    ).toBeUndefined();
  });
});

describe("serializeTierModelConfigs", () => {
  const tiers: ComplexityTiers = { SIMPLE: ["mini"], MEDIUM: ["opus"], COMPLEX: ["opus"], REASONING: [] };

  it("emits the sibling wire shape per tier and model", () => {
    expect(
      serializeTierModelConfigs(tiers, {
        MEDIUM: { opus: { reasoning_effort: "medium" } },
        COMPLEX: { opus: { reasoning_effort: "high" } },
      }),
    ).toEqual({
      MEDIUM: [{ model_name: "opus", litellm_params: { reasoning_effort: "medium" } }],
      COMPLEX: [{ model_name: "opus", litellm_params: { reasoning_effort: "high" } }],
    });
  });

  it("prunes params for a model no longer selected in the tier", () => {
    expect(
      serializeTierModelConfigs(tiers, { MEDIUM: { "removed-model": { reasoning_effort: "low" } } }),
    ).toBeUndefined();
  });

  // Params authored in config.yaml alongside reasoning_effort must survive an edit round-trip.
  it("carries params keys this editor has no control for", () => {
    expect(
      serializeTierModelConfigs(tiers, { MEDIUM: { opus: { reasoning_effort: "medium", max_tokens: 512 } } }),
    ).toEqual({
      MEDIUM: [{ model_name: "opus", litellm_params: { reasoning_effort: "medium", max_tokens: 512 } }],
    });
  });

  // This modal renders only the four built-in tiers; params stored under an operator-defined tier
  // must pass through rather than being dropped the moment the key became managed.
  it("passes tiers this editor does not render through untouched", () => {
    expect(serializeTierModelConfigs(tiers, { DEEP_RESEARCH: { opus: { reasoning_effort: "xhigh" } } })).toEqual({
      DEEP_RESEARCH: [{ model_name: "opus", litellm_params: { reasoning_effort: "xhigh" } }],
    });
  });

  it("round-trips what hydration produced", () => {
    const stored = { MEDIUM: [{ model_name: "opus", litellm_params: { reasoning_effort: "medium" } }] };
    expect(serializeTierModelConfigs(tiers, hydrateTierModelParams(tiers, stored))).toEqual(stored);
  });

  it("serializes to undefined when nothing is set", () => {
    expect(serializeTierModelConfigs(tiers, undefined)).toBeUndefined();
    expect(serializeTierModelConfigs(tiers, { MEDIUM: {} })).toBeUndefined();
  });
});

describe("setTierModelReasoningEffort", () => {
  it("sets an effort for a tier and model", () => {
    expect(setTierModelReasoningEffort(undefined, "MEDIUM", "opus", "medium")).toEqual({
      MEDIUM: { opus: { reasoning_effort: "medium" } },
    });
  });

  it("unsetting removes the key and collapses empties back to undefined", () => {
    const set = setTierModelReasoningEffort(undefined, "MEDIUM", "opus", "medium");
    expect(setTierModelReasoningEffort(set, "MEDIUM", "opus", undefined)).toBeUndefined();
  });

  it("unsetting the effort keeps params keys it does not own", () => {
    expect(
      setTierModelReasoningEffort(
        { MEDIUM: { opus: { reasoning_effort: "medium", max_tokens: 512 } } },
        "MEDIUM",
        "opus",
        undefined,
      ),
    ).toEqual({ MEDIUM: { opus: { max_tokens: 512 } } });
  });

  it("leaves other tiers and models alone", () => {
    expect(
      setTierModelReasoningEffort({ COMPLEX: { opus: { reasoning_effort: "high" } } }, "MEDIUM", "opus", "low"),
    ).toEqual({
      COMPLEX: { opus: { reasoning_effort: "high" } },
      MEDIUM: { opus: { reasoning_effort: "low" } },
    });
  });
});

describe("pruneTierModelParams", () => {
  it("drops params for models deselected from the tier", () => {
    expect(
      pruneTierModelParams({ MEDIUM: { opus: { reasoning_effort: "medium" } } }, "MEDIUM", ["mini"]),
    ).toBeUndefined();
  });

  it("keeps params for models still selected", () => {
    const current = { MEDIUM: { opus: { reasoning_effort: "medium" } } };
    expect(pruneTierModelParams(current, "MEDIUM", ["opus", "mini"])).toEqual(current);
  });

  it("returns the input unchanged when the tier holds no params", () => {
    const current = { COMPLEX: { opus: { reasoning_effort: "high" } } };
    expect(pruneTierModelParams(current, "MEDIUM", [])).toBe(current);
  });
});
