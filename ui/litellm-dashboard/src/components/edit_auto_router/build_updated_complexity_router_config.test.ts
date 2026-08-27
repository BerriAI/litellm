import { describe, expect, it } from "vitest";

import {
  MANAGED_COMPLEXITY_ROUTER_KEYS,
  buildUpdatedComplexityRouterConfig,
  hydrateComplexityRouterConfig,
  type KeywordMatchingState,
} from "./edit_auto_router_modal";

const STORED = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: [], REASONING: [] },
  classifier_type: "heuristic",
  keyword_tier_rules: [{ keywords: ["invoice", "refund"], tier: "MEDIUM" }],
  escalation_keywords: ["urgent", "outage"],
  semantic_keyword_matching: true,
  embedding_model: "voyage-4-large",
  match_threshold: 0.72,
  // A key no UI control owns; it must survive every save untouched.
  some_future_backend_key: { nested: true },
};

const FORM_VALUE = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: [], REASONING: [] },
  classifier_type: "heuristic" as const,
};

const hydratedState: KeywordMatchingState = {
  keywordTierRules: [{ id: "stored-0", keywords: ["invoice", "refund"], tier: "MEDIUM" }],
  escalationKeywords: ["urgent", "outage"],
  semanticMatchingEnabled: true,
  embeddingModel: "voyage-4-large",
  matchThreshold: 0.72,
};

describe("buildUpdatedComplexityRouterConfig keyword matching", () => {
  it("round-trips an untouched edit without changing any keyword-matching value", () => {
    // Opening the modal hydrates state from STORED; saving with nothing changed must be a
    // no-op. These keys are now MANAGED, so a hydration bug silently wipes them.
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, hydratedState);

    expect(result.keyword_tier_rules).toEqual([{ keywords: ["invoice", "refund"], tier: "MEDIUM" }]);
    expect(result.escalation_keywords).toEqual(["urgent", "outage"]);
    expect(result.semantic_keyword_matching).toBe(true);
    expect(result.embedding_model).toBe("voyage-4-large");
    expect(result.match_threshold).toBe(0.72);
  });

  it("preserves keys no control owns", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, hydratedState);
    expect(result.some_future_backend_key).toEqual({ nested: true });
  });

  it("persists an edited keyword rule", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, {
      ...hydratedState,
      keywordTierRules: [{ id: "stored-0", keywords: ["chargeback"], tier: "COMPLEX" }],
    });

    expect(result.keyword_tier_rules).toEqual([{ keywords: ["chargeback"], tier: "COMPLEX" }]);
  });

  // getKeywordTierRulesError blocks this save, so the builder never runs on a real edit. Keeping
  // the rule here means that if a caller ever reaches it anyway, the stored rules are replaced by
  // something the backend rejects out loud rather than by silence that reads as a clean save.
  it("keeps a rule left empty rather than quietly dropping the caller's row", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, {
      ...hydratedState,
      keywordTierRules: [{ id: "new-1", keywords: ["   "], tier: "SIMPLE" }],
    });

    expect(result.keyword_tier_rules).toEqual([{ keywords: [], tier: "SIMPLE" }]);
  });

  it("removes the semantic trio when the toggle is turned off", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, {
      ...hydratedState,
      semanticMatchingEnabled: false,
    });

    expect(result.semantic_keyword_matching).toBeUndefined();
    expect(result.embedding_model).toBeUndefined();
    expect(result.match_threshold).toBeUndefined();
  });

  it("carries stored keyword matching through untouched when the caller owns no such state", () => {
    // Any caller that does not render these controls must not have its values dropped just
    // because the keys are listed as managed.
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE);

    expect(result.keyword_tier_rules).toEqual([{ keywords: ["invoice", "refund"], tier: "MEDIUM" }]);
    expect(result.escalation_keywords).toEqual(["urgent", "outage"]);
    expect(result.semantic_keyword_matching).toBe(true);
    expect(result.embedding_model).toBe("voyage-4-large");
    expect(result.match_threshold).toBe(0.72);
  });
});

const STORED_LLM = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: [], REASONING: [] },
  classifier_type: "llm",
  classifier_llm_config: { model: "gpt-4o-mini", timeout_ms: 3000 },
  classifier_context_window_size: 5,
  classifier_context_per_turn_chars: 300,
};

describe("buildUpdatedComplexityRouterConfig classifier context window", () => {
  it("round-trips an untouched edit without changing the classifier context values", () => {
    const formValue = {
      tiers: STORED_LLM.tiers,
      classifier_type: "llm" as const,
      classifier_llm_config: STORED_LLM.classifier_llm_config,
      classifier_context_window_size: 5,
      classifier_context_per_turn_chars: 300,
    };
    const result = buildUpdatedComplexityRouterConfig(STORED_LLM, formValue);

    expect(result.classifier_context_window_size).toBe(5);
    expect(result.classifier_context_per_turn_chars).toBe(300);
  });

  it("persists an edited classifier context window size", () => {
    const formValue = {
      tiers: STORED_LLM.tiers,
      classifier_type: "llm" as const,
      classifier_llm_config: STORED_LLM.classifier_llm_config,
      classifier_context_window_size: 10,
    };
    const result = buildUpdatedComplexityRouterConfig(STORED_LLM, formValue);

    expect(result.classifier_context_window_size).toBe(10);
  });

  it("carries a stored per-turn cap through untouched now that no control sets it", () => {
    // The modal stopped rendering a per-turn control, so the key left MANAGED_COMPLEXITY_ROUTER_KEYS.
    // Had it stayed managed, every open-and-save would have silently dropped an operator's cap.
    const formValue = {
      tiers: STORED_LLM.tiers,
      classifier_type: "llm" as const,
      classifier_llm_config: STORED_LLM.classifier_llm_config,
      classifier_context_window_size: 10,
    };
    const result = buildUpdatedComplexityRouterConfig(STORED_LLM, formValue);

    expect(result.classifier_context_per_turn_chars).toBe(300);
  });

  it("omits classifier context fields when classifier_type is heuristic even if values linger in state", () => {
    const formValue = {
      tiers: STORED_LLM.tiers,
      classifier_type: "heuristic" as const,
      classifier_context_window_size: 5,
      classifier_context_budget_chars: 4000,
    };
    const result = buildUpdatedComplexityRouterConfig(STORED_LLM, formValue);

    expect(result.classifier_context_window_size).toBeUndefined();
    expect(result.classifier_context_budget_chars).toBeUndefined();
  });

  it("does not resurrect a stale stored classifier_context_window_size once the form's own value is unset", () => {
    // classifier_context_window_size is a MANAGED key: the form's value must win over whatever
    // is still sitting in the stored config, never fall back to it through preservedConfig.
    const formValue = {
      tiers: STORED_LLM.tiers,
      classifier_type: "llm" as const,
      classifier_llm_config: STORED_LLM.classifier_llm_config,
    };
    const result = buildUpdatedComplexityRouterConfig(STORED_LLM, formValue);

    expect(result.classifier_context_window_size).toBeUndefined();
  });
});

const STORED_ASSISTANT_CTX = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: [], REASONING: [] },
  classifier_type: "llm",
  classifier_llm_config: { model: "gpt-4o-mini", timeout_ms: 3000 },
  classifier_context_include_assistant_turns: true,
};

describe("buildUpdatedComplexityRouterConfig assistant turns", () => {
  const formBase = {
    tiers: STORED_ASSISTANT_CTX.tiers,
    classifier_type: "llm" as const,
    classifier_llm_config: STORED_ASSISTANT_CTX.classifier_llm_config,
  };

  it("round-trips an untouched edit without changing the value", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED_ASSISTANT_CTX, {
      ...formBase,
      classifier_context_include_assistant_turns: true,
    });
    expect(result.classifier_context_include_assistant_turns).toBe(true);
  });

  it("persists turning assistant turns back off", () => {
    // The off case is the one a preserved-config fallback would silently lose, since false and
    // "absent" look alike to a truthiness check.
    const result = buildUpdatedComplexityRouterConfig(STORED_ASSISTANT_CTX, {
      ...formBase,
      classifier_context_include_assistant_turns: false,
    });
    expect(result.classifier_context_include_assistant_turns).toBe(false);
  });

  it("omits it when classifier_type is heuristic even if a value lingers in state", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED_ASSISTANT_CTX, {
      tiers: STORED_ASSISTANT_CTX.tiers,
      classifier_type: "heuristic" as const,
      classifier_context_include_assistant_turns: true,
    });
    expect(result.classifier_context_include_assistant_turns).toBeUndefined();
  });

  it("does not resurrect a stale stored value once the form's own value is unset", () => {
    // A MANAGED key: the form wins over the stored config, never falls back to it.
    const result = buildUpdatedComplexityRouterConfig(STORED_ASSISTANT_CTX, formBase);
    expect(result.classifier_context_include_assistant_turns).toBeUndefined();
  });
});

describe("buildUpdatedComplexityRouterConfig session affinity", () => {
  it("writes session_affinity=false when the toggle is off", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, { ...FORM_VALUE, session_affinity: false });
    expect(result.session_affinity).toBe(false);
  });

  it("writes session_affinity=true when the toggle is on", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, { ...FORM_VALUE, session_affinity: true });
    expect(result.session_affinity).toBe(true);
  });

  it("re-asserts the backend's off-by-default when the form value is absent, rather than dropping the key", () => {
    const result = buildUpdatedComplexityRouterConfig({ ...STORED, session_affinity: true }, FORM_VALUE);
    expect(result.session_affinity).toBe(false);
  });

  it("stops a stored session_affinity=true from surviving a save that turned the toggle back off", () => {
    const result = buildUpdatedComplexityRouterConfig(
      { ...STORED, session_affinity: true },
      { ...FORM_VALUE, session_affinity: false },
    );
    expect(result.session_affinity).toBe(false);
  });
});

describe("buildUpdatedComplexityRouterConfig deployment affinity", () => {
  it("writes deployment_affinity=false when the toggle is off", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, { ...FORM_VALUE, deployment_affinity: false });
    expect(result.deployment_affinity).toBe(false);
  });

  it("writes deployment_affinity=true when the toggle is on", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, { ...FORM_VALUE, deployment_affinity: true });
    expect(result.deployment_affinity).toBe(true);
  });

  it("re-asserts the backend's on-by-default when the form value is absent, rather than dropping the key", () => {
    const result = buildUpdatedComplexityRouterConfig({ ...STORED, deployment_affinity: false }, FORM_VALUE);
    expect(result.deployment_affinity).toBe(true);
  });

  it("stops a stored deployment_affinity=false from surviving a save that turned the toggle back on", () => {
    const result = buildUpdatedComplexityRouterConfig(
      { ...STORED, deployment_affinity: false },
      { ...FORM_VALUE, deployment_affinity: true },
    );
    expect(result.deployment_affinity).toBe(true);
  });
});

describe("buildUpdatedComplexityRouterConfig tier labels", () => {
  const RENAMED = { ...STORED, tier_labels: { SIMPLE: "Cheap", REASONING: "Deep" } };

  it("round-trips stored labels through an untouched edit", () => {
    const result = buildUpdatedComplexityRouterConfig(RENAMED, {
      ...FORM_VALUE,
      tier_labels: { SIMPLE: "Cheap", REASONING: "Deep" },
    });
    expect(result.tier_labels).toEqual({ SIMPLE: "Cheap", REASONING: "Deep" });
  });

  it("persists a renamed tier", () => {
    const result = buildUpdatedComplexityRouterConfig(RENAMED, {
      ...FORM_VALUE,
      tier_labels: { SIMPLE: "Budget", REASONING: "Deep" },
    });
    expect(result.tier_labels).toEqual({ SIMPLE: "Budget", REASONING: "Deep" });
  });

  it("drops the key when every label is cleared back to the default", () => {
    const result = buildUpdatedComplexityRouterConfig(RENAMED, { ...FORM_VALUE, tier_labels: {} });
    expect(result.tier_labels).toBeUndefined();
    expect("tier_labels" in result).toBe(false);
  });

  it("leaves an unrenamed router without the key", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE);
    expect("tier_labels" in result).toBe(false);
  });

  it("keeps the tiers keys canonical alongside a rename", () => {
    const result = buildUpdatedComplexityRouterConfig(RENAMED, {
      ...FORM_VALUE,
      tier_labels: { SIMPLE: "Cheap" },
    });
    expect(Object.keys(result.tiers as Record<string, unknown>)).toEqual(["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]);
  });
});

describe("buildUpdatedComplexityRouterConfig scorer knobs", () => {
  const BOUNDARIES = { simple_medium: 0.22, medium_complex: 0.44, complex_reasoning: 0.66 };
  const STORED_WITH_KNOBS = { ...STORED, tier_boundaries: BOUNDARIES };
  const HYDRATED = { ...FORM_VALUE, tier_boundaries: BOUNDARIES };

  it("round-trips explicit stored knobs through an untouched edit", () => {
    // These keys are MANAGED now, so the stored copy is dropped before the rebuild and only a faithful
    // hydration puts them back. A regression here silently resets a tuned router.
    expect(buildUpdatedComplexityRouterConfig(STORED_WITH_KNOBS, HYDRATED).tier_boundaries).toEqual(BOUNDARIES);
  });

  it("drops a stored knob when the operator resets it, instead of preserving the old value", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED_WITH_KNOBS, FORM_VALUE);

    expect(result).not.toHaveProperty("tier_boundaries");
    expect(result.some_future_backend_key).toEqual({ nested: true });
  });

  it("never invents knobs for a router that never had them", () => {
    expect(buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE)).not.toHaveProperty("tier_boundaries");
  });

  // 0 is an unconditional reasoning override. Treating it as unset here would quietly retune the router
  // back to tracking simple_medium on the next save.
  it("round-trips a stored reasoning override floor of 0", () => {
    const result = buildUpdatedComplexityRouterConfig(
      { ...STORED, reasoning_override_min_score: 0 },
      { ...FORM_VALUE, reasoning_override_min_score: 0 },
    );
    expect(result.reasoning_override_min_score).toBe(0);
  });

  it("writes a newly set reasoning override floor over the stored one", () => {
    const result = buildUpdatedComplexityRouterConfig(
      { ...STORED, reasoning_override_min_score: 0 },
      { ...FORM_VALUE, reasoning_override_min_score: 0.5 },
    );
    expect(result.reasoning_override_min_score).toBe(0.5);
  });

  it("drops a stored reasoning override floor when the operator resets it", () => {
    const result = buildUpdatedComplexityRouterConfig({ ...STORED, reasoning_override_min_score: 0.5 }, FORM_VALUE);

    expect(result).not.toHaveProperty("reasoning_override_min_score");
    expect(result.some_future_backend_key).toEqual({ nested: true });
  });

  it("drops the reasoning override floor on a router whose scorer never runs", () => {
    const neverScores = {
      ...FORM_VALUE,
      reasoning_override_min_score: 0,
      classifier_type: "llm" as const,
      classifier_fallback: "default_model" as const,
    };
    const result = buildUpdatedComplexityRouterConfig({ ...STORED, reasoning_override_min_score: 0 }, neverScores);
    expect(result).not.toHaveProperty("reasoning_override_min_score");
  });
});

describe("buildUpdatedComplexityRouterConfig plan-mode minimum tier", () => {
  it("round-trips a stored tier through an untouched open-and-save", () => {
    const result = buildUpdatedComplexityRouterConfig(
      { ...STORED, plan_mode_min_tier: "COMPLEX" },
      { ...FORM_VALUE, plan_mode_min_tier: "COMPLEX" },
    );
    expect(result.plan_mode_min_tier).toBe("COMPLEX");
  });

  it("stops a stored tier from surviving a save that turned the override off", () => {
    const result = buildUpdatedComplexityRouterConfig({ ...STORED, plan_mode_min_tier: "COMPLEX" }, FORM_VALUE);
    expect(result).not.toHaveProperty("plan_mode_min_tier");
  });

  it("writes a newly selected tier over the stored one", () => {
    const result = buildUpdatedComplexityRouterConfig(
      { ...STORED, plan_mode_min_tier: "COMPLEX" },
      { ...FORM_VALUE, plan_mode_min_tier: "MEDIUM" },
    );
    expect(result.plan_mode_min_tier).toBe("MEDIUM");
  });
});

describe("buildUpdatedComplexityRouterConfig tier model params", () => {
  const storedWithParams = {
    ...STORED,
    tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["opus"], COMPLEX: ["opus"], REASONING: [] },
    tier_model_configs: {
      MEDIUM: [{ model_name: "opus", litellm_params: { reasoning_effort: "medium" } }],
      COMPLEX: [{ model_name: "opus", litellm_params: { reasoning_effort: "high" } }],
    },
  };
  const formValueWithParams = {
    ...FORM_VALUE,
    tiers: storedWithParams.tiers,
    tier_model_params: {
      MEDIUM: { opus: { reasoning_effort: "medium" } },
      COMPLEX: { opus: { reasoning_effort: "high" } },
    },
  };

  it("round-trips hydrated params on an untouched save", () => {
    const result = buildUpdatedComplexityRouterConfig(storedWithParams, formValueWithParams, undefined, hydratedState);
    expect(result.tier_model_configs).toEqual(storedWithParams.tier_model_configs);
  });

  // tier_model_configs is managed now that this modal renders a control for it. Before that, the
  // stale stored key was carried through, so clearing the last effort could never persist.
  it("drops the stored key entirely when the operator unsets every effort", () => {
    const result = buildUpdatedComplexityRouterConfig(
      storedWithParams,
      { ...formValueWithParams, tier_model_params: undefined },
      undefined,
      hydratedState,
    );
    expect(result).not.toHaveProperty("tier_model_configs");
  });

  it("drops params for a model removed from its tier", () => {
    const result = buildUpdatedComplexityRouterConfig(
      storedWithParams,
      {
        ...formValueWithParams,
        tiers: { ...storedWithParams.tiers, COMPLEX: ["gpt-4o-mini"] },
      },
      undefined,
      hydratedState,
    );
    expect(result.tier_model_configs).toEqual({
      MEDIUM: [{ model_name: "opus", litellm_params: { reasoning_effort: "medium" } }],
    });
  });

  it("emits no tier_model_configs for a config that never had params", () => {
    const result = buildUpdatedComplexityRouterConfig(STORED, FORM_VALUE, undefined, hydratedState);
    expect(result).not.toHaveProperty("tier_model_configs");
  });
});

describe("managed keys survive an untouched open-and-save", () => {
  // Every managed key is rewritten from form state on save, so one the hydrator forgets is silently
  // dropped from the saved config. This config sets each managed key to a value that actually
  // applies, so an untouched open-and-save must return every one of them.
  const STORED_ALL_MANAGED: Record<string, unknown> = {
    tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o"], COMPLEX: ["opus"], REASONING: ["o1"] },
    tier_model_configs: { REASONING: [{ model_name: "o1", litellm_params: { reasoning_effort: "high" } }] },
    default_model: "gpt-4o",
    plan_mode_min_tier: "COMPLEX",
    tier_labels: { SIMPLE: "Cheap" },
    classifier_type: "heuristic_first",
    heuristic_first_max_tier: "SIMPLE",
    classifier_llm_config: { model: "gpt-4o-mini", timeout_ms: 3000 },
    classifier_context_window_size: 5,
    classifier_context_budget_chars: 4000,
    classifier_context_include_assistant_turns: true,
    classifier_fallback: "default_model",
    session_affinity: true,
    deployment_affinity: false,
    adaptive: true,
    adaptive_weights: { quality: 0.4, cost: 0.6 },
    tier_distance_penalty: 0.25,
    adaptive_eligible: "all",
    return_raw_model_name: true,
    tier_boundaries: { simple_medium: 0.2, medium_complex: 0.4, complex_reasoning: 0.7 },
    token_thresholds: { simple: 20, complex: 500 },
    dimension_weights: { tokenCount: 0.1 },
    reasoning_override_min_score: 0.3,
  };

  it("carries every managed key through hydrate then save", () => {
    const hydrated = hydrateComplexityRouterConfig(STORED_ALL_MANAGED, undefined);
    const saved = buildUpdatedComplexityRouterConfig(STORED_ALL_MANAGED, hydrated);

    const dropped = [...MANAGED_COMPLEXITY_ROUTER_KEYS].filter((key) => saved[key] === undefined);
    expect(dropped).toEqual([]);
  });

  it("round-trips the heuristic_first threshold, which save requires and the backend rejects without", () => {
    const hydrated = hydrateComplexityRouterConfig(STORED_ALL_MANAGED, undefined);
    expect(hydrated.heuristic_first_max_tier).toBe("SIMPLE");
    expect(buildUpdatedComplexityRouterConfig(STORED_ALL_MANAGED, hydrated).heuristic_first_max_tier).toBe("SIMPLE");
  });
});
