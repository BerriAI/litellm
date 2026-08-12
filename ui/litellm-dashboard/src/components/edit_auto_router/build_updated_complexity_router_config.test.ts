import { describe, expect, it } from "vitest";

import { buildUpdatedComplexityRouterConfig, type KeywordMatchingState } from "./edit_auto_router_modal";

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

  it("persists an edited classifier context window size and per-turn char limit", () => {
    const formValue = {
      tiers: STORED_LLM.tiers,
      classifier_type: "llm" as const,
      classifier_llm_config: STORED_LLM.classifier_llm_config,
      classifier_context_window_size: 10,
      classifier_context_per_turn_chars: 500,
    };
    const result = buildUpdatedComplexityRouterConfig(STORED_LLM, formValue);

    expect(result.classifier_context_window_size).toBe(10);
    expect(result.classifier_context_per_turn_chars).toBe(500);
  });

  it("omits classifier context fields when classifier_type is heuristic even if values linger in state", () => {
    const formValue = {
      tiers: STORED_LLM.tiers,
      classifier_type: "heuristic" as const,
      classifier_context_window_size: 5,
      classifier_context_per_turn_chars: 300,
    };
    const result = buildUpdatedComplexityRouterConfig(STORED_LLM, formValue);

    expect(result.classifier_context_window_size).toBeUndefined();
    expect(result.classifier_context_per_turn_chars).toBeUndefined();
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
    expect(result.classifier_context_per_turn_chars).toBeUndefined();
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
