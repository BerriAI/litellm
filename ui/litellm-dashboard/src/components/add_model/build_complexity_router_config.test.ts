import {
  buildComplexityRouterConfig,
  getPlanModeTierError,
  normalizeClassifierLlmConfig,
  getKeywordTierRulesError,
  getClassifierModelError,
  getMissingTiersError,
  getSemanticConfigError,
  getTierLabelsError,
  hydrateTierLabels,
  BuildComplexityRouterConfigParams,
} from "./build_complexity_router_config";
import { activeTierRows } from "./tier_rows";

const tiers = {
  SIMPLE: ["gpt-4o-mini"],
  MEDIUM: ["gpt-4o"],
  COMPLEX: ["claude-sonnet-4"],
  REASONING: ["o1-preview"],
};

const baseParams: BuildComplexityRouterConfigParams = {
  tiers,
  tierLabels: undefined,
  classifierType: "heuristic",
  classifierLlmConfig: undefined,
  classifierContextWindowSize: undefined,
  classifierContextBudgetChars: undefined,
  classifierContextIncludeAssistantTurns: undefined,
  classifierFallback: undefined,
  sessionAffinity: false,
  deploymentAffinity: true,
  customTechnicalKeywords: [],
  keywordTierRules: [],
  semanticMatchingEnabled: false,
  embeddingModel: undefined,
  matchThreshold: 0.5,
  escalationKeywords: ["LITELLM ESCALATE"],
  adaptive: false,
  adaptiveWeights: { quality: 0.3, cost: 0.7 },
  tierDistancePenalty: 0.5,
  adaptiveEligible: "all",
  returnRawModelName: false,
};

describe("buildComplexityRouterConfig", () => {
  it("emits tiers, classifier_type, and escalation_keywords when nothing else is configured", () => {
    const config = buildComplexityRouterConfig(baseParams);
    expect(config).toEqual({
      tiers,
      classifier_type: "heuristic",
      session_affinity: false,
      deployment_affinity: true,
      escalation_keywords: ["LITELLM ESCALATE"],
    });
  });

  it("trims escalation keywords and drops blank entries", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      escalationKeywords: [" LITELLM ESCALATE ", "", "  ", "MAKE IT BETTER"],
    });
    expect(config.escalation_keywords).toEqual(["LITELLM ESCALATE", "MAKE IT BETTER"]);
  });

  it("emits an empty escalation_keywords list so clearing the field disables escalation", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, escalationKeywords: [] });
    expect(config.escalation_keywords).toEqual([]);
  });

  it("passes through a tier configured with more than one model as a pool", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      tiers: { ...tiers, SIMPLE: ["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5"] },
    });
    expect(config.tiers.SIMPLE).toEqual(["gpt-4o-mini", "gpt-4o", "claude-haiku-4-5"]);
  });

  it("includes classifier_llm_config only when classifier_type is llm", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      classifierType: "llm",
      classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
    });
    expect(config.classifier_type).toBe("llm");
    expect(config.classifier_llm_config).toEqual({ model: "gpt-4o-mini", timeout_ms: 3000 });
  });

  it("omits classifier_llm_config when classifier_type is heuristic even if config lingers in state", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      classifierType: "heuristic",
      classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
    });
    expect(config.classifier_llm_config).toBeUndefined();
  });

  it("includes classifier_context_window_size and classifier_context_budget_chars only when classifier_type is llm", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      classifierType: "llm",
      classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
      classifierContextWindowSize: 5,
      classifierContextBudgetChars: 4000,
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.classifier_context_window_size).toBe(5);
    expect(config.classifier_context_budget_chars).toBe(4000);
  });

  it("omits classifier_context_window_size and classifier_context_budget_chars when classifier_type is heuristic even if values linger in state", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      classifierType: "heuristic",
      classifierContextWindowSize: 5,
      classifierContextBudgetChars: 4000,
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.classifier_context_window_size).toBeUndefined();
    expect(config.classifier_context_budget_chars).toBeUndefined();
  });

  it("omits classifier_context_window_size and classifier_context_budget_chars when classifier_type is llm but neither was set, leaving the backend default", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      classifierType: "llm",
      classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
    });
    expect(config.classifier_context_window_size).toBeUndefined();
    expect(config.classifier_context_budget_chars).toBeUndefined();
  });

  it("allows classifier_context_window_size of 0, distinct from unset, to send no prior-turn context", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      classifierType: "llm",
      classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
      classifierContextWindowSize: 0,
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.classifier_context_window_size).toBe(0);
  });

  it("sends keyword_tier_rules with their per-tier targeting preserved (not flattened)", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      customTechnicalKeywords: ["udp"],
      keywordTierRules: [
        { id: "r1", keywords: ["deploy to k8s"], tier: "REASONING" },
        { id: "r2", keywords: ["invoice", "refund"], tier: "SIMPLE" },
      ],
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.keyword_tier_rules).toEqual([
      { keywords: ["deploy to k8s"], tier: "REASONING" },
      { keywords: ["invoice", "refund"], tier: "SIMPLE" },
    ]);
    // custom technical keywords stay their own list, not merged with rule keywords
    expect(config.custom_technical_keywords).toEqual(["udp"]);
    expect(config.semantic_keyword_matching).toBeUndefined();
  });

  it("includes semantic fields only when semantic matching is enabled", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [{ id: "r1", keywords: ["k8s"], tier: "REASONING" }],
      semanticMatchingEnabled: true,
      embeddingModel: "openai/text-embedding-3-small",
      matchThreshold: 0.42,
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.semantic_keyword_matching).toBe(true);
    expect(config.embedding_model).toBe("openai/text-embedding-3-small");
    expect(config.match_threshold).toBe(0.42);
  });

  it("omits semantic fields when the toggle is off even if an embedding model lingers in state", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [{ id: "r1", keywords: ["k8s"], tier: "REASONING" }],
      semanticMatchingEnabled: false,
      embeddingModel: "openai/text-embedding-3-small",
      matchThreshold: 0.42,
    };
    const config = buildComplexityRouterConfig(params);
    expect(config.semantic_keyword_matching).toBeUndefined();
    expect(config.embedding_model).toBeUndefined();
    expect(config.match_threshold).toBeUndefined();
  });

  it("omits empty optional lists", () => {
    const config = buildComplexityRouterConfig(baseParams);
    expect(config.custom_technical_keywords).toBeUndefined();
    expect(config.keyword_tier_rules).toBeUndefined();
  });

  it("trims keywords but keeps rules left empty, so a dropped row can never pass for a saved one", () => {
    const params: BuildComplexityRouterConfigParams = {
      ...baseParams,
      keywordTierRules: [
        { id: "r1", keywords: [" deploy to k8s ", "", "  "], tier: "REASONING" },
        { id: "r2", keywords: [], tier: "COMPLEX" }, // seeded by "Add keyword rule", never filled
        { id: "r3", keywords: ["   "], tier: "SIMPLE" }, // whitespace only
      ],
    };
    const config = buildComplexityRouterConfig(params);
    // getKeywordTierRulesError blocks this submit; r2 and r3 survive here so the backend rejects
    // them loudly rather than the caller's rows vanishing on a successful save.
    expect(config.keyword_tier_rules).toEqual([
      { keywords: ["deploy to k8s"], tier: "REASONING" },
      { keywords: [], tier: "COMPLEX" },
      { keywords: [], tier: "SIMPLE" },
    ]);
  });

  it("omits adaptive fields when adaptive is disabled even if weights linger in state", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      adaptive: false,
      adaptiveWeights: { quality: 0.9, cost: 0.1 },
      tierDistancePenalty: 2,
      adaptiveEligible: "classified_tier",
    });
    expect(config.adaptive).toBeUndefined();
    expect(config.adaptive_weights).toBeUndefined();
    expect(config.tier_distance_penalty).toBeUndefined();
    expect(config.adaptive_eligible).toBeUndefined();
  });

  it("omits return_raw_model_name when disabled", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, returnRawModelName: false });
    expect(config.return_raw_model_name).toBeUndefined();
  });

  it("writes session_affinity=true so turning the toggle on overrides the backend's off-by-default", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, sessionAffinity: true });
    expect(config.session_affinity).toBe(true);
  });

  it("writes session_affinity explicitly when off, so the stored config never relies on the backend default", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, sessionAffinity: false });
    expect(config.session_affinity).toBe(false);
  });

  it("includes return_raw_model_name when enabled", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, returnRawModelName: true });
    expect(config.return_raw_model_name).toBe(true);
  });

  it("includes tier_distance_penalty when adaptive is enabled with eligible='all'", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      adaptive: true,
      adaptiveWeights: { quality: 0.6, cost: 0.4 },
      tierDistancePenalty: 0.75,
      adaptiveEligible: "all",
    });
    expect(config.adaptive).toBe(true);
    expect(config.adaptive_weights).toEqual({ quality: 0.6, cost: 0.4 });
    expect(config.tier_distance_penalty).toBe(0.75);
    expect(config.adaptive_eligible).toBe("all");
  });

  it("omits tier_distance_penalty when eligible='classified_tier', since the penalty doesn't apply there", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      adaptive: true,
      adaptiveWeights: { quality: 0.6, cost: 0.4 },
      tierDistancePenalty: 0.75,
      adaptiveEligible: "classified_tier",
    });
    expect(config.adaptive).toBe(true);
    expect(config.adaptive_eligible).toBe("classified_tier");
    expect(config.tier_distance_penalty).toBeUndefined();
  });
});

describe("getMissingTiersError", () => {
  it("returns null when all four tiers have a model", () => {
    expect(getMissingTiersError(activeTierRows({ tiers: tiers }))).toBeNull();
  });

  it("names the specific missing tier when only one is blank", () => {
    expect(getMissingTiersError(activeTierRows({ tiers: { ...tiers, REASONING: [] } }))).toBe(
      "Select a model for the following tier(s): REASONING",
    );
  });

  it("names multiple missing tiers in SIMPLE/MEDIUM/COMPLEX/REASONING order", () => {
    expect(getMissingTiersError(activeTierRows({ tiers: { ...tiers, SIMPLE: [], REASONING: [] } }))).toBe(
      "Select a model for the following tier(s): SIMPLE, REASONING",
    );
  });

  it("names all four tiers when none are filled", () => {
    const noTiers = { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] };
    expect(getMissingTiersError(activeTierRows({ tiers: noTiers }))).toBe(
      "Select a model for the following tier(s): SIMPLE, MEDIUM, COMPLEX, REASONING",
    );
  });

  it("treats a tier with more than one model as filled", () => {
    expect(getMissingTiersError(activeTierRows({ tiers: { ...tiers, SIMPLE: ["gpt-4o-mini", "gpt-4o"] } }))).toBeNull();
  });
});

describe("getSemanticConfigError", () => {
  const rule = { id: "r1", keywords: ["k8s"], tier: "REASONING" as const };

  it("returns null when semantic matching is disabled (even with gaps)", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: false, embeddingModel: undefined, keywordTierRules: [] }),
    ).toBeNull();
  });

  it("errors when enabled without an embedding model", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: undefined, keywordTierRules: [rule] }),
    ).toMatch(/embedding model/i);
  });

  it("errors when enabled with an embedding model but no keyword tier rules", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: "voyage-3-5", keywordTierRules: [] }),
    ).toMatch(/keyword tier rule/i);
  });

  it("returns null when enabled with both an embedding model and rules", () => {
    expect(
      getSemanticConfigError({ semanticMatchingEnabled: true, embeddingModel: "voyage-3-5", keywordTierRules: [rule] }),
    ).toBeNull();
  });
});

describe("getKeywordTierRulesError", () => {
  it("returns null when every rule carries a keyword", () => {
    expect(
      getKeywordTierRulesError(
        [
          { id: "r1", keywords: ["invoice"], tier: "MEDIUM" },
          { id: "r2", keywords: ["deploy to k8s"], tier: "REASONING" },
        ],
        activeTierRows({ tiers }),
      ),
    ).toBeNull();
  });

  it("returns null when there are no rules at all, since the section is optional", () => {
    expect(getKeywordTierRulesError([], activeTierRows({ tiers }))).toBeNull();
  });

  // The whole point of the ticket: the semantic toggle is off by default, and an unfilled row
  // used to be discarded silently on an otherwise successful create.
  it("rejects a row left empty while semantic matching is off", () => {
    expect(getKeywordTierRulesError([{ id: "r1", keywords: [], tier: "COMPLEX" }], activeTierRows({ tiers }))).toBe(
      "Add at least one keyword to keyword rule(s): 1",
    );
  });

  it.each([
    ["whitespace only", ["   "]],
    ["blank strings, as an unfilled row between filled ones leaves behind", ["", " ", ""]],
  ])("treats %s as empty rather than as a keyword", (_label, keywords) => {
    expect(getKeywordTierRulesError([{ id: "r1", keywords, tier: "SIMPLE" }], activeTierRows({ tiers }))).toMatch(
      /keyword rule\(s\): 1/,
    );
  });

  // Row numbers have to survive rules that are fine, or the message points at the wrong input.
  it("names each offending row by its position among all rules", () => {
    expect(
      getKeywordTierRulesError([
        { id: "r1", keywords: ["invoice"], tier: "MEDIUM" },
        { id: "r2", keywords: [], tier: "COMPLEX" },
        { id: "r3", keywords: ["billing"], tier: "SIMPLE" },
        { id: "r4", keywords: ["  "], tier: "REASONING" },
      ]),
    ).toBe("Add at least one keyword to keyword rule(s): 2, 4");
  });

  it("keeps a keyword whose surrounding whitespace is the only thing trimmed", () => {
    expect(
      getKeywordTierRulesError([{ id: "r1", keywords: ["  invoice  "], tier: "MEDIUM" }], activeTierRows({ tiers })),
    ).toBeNull();
  });
});

describe("buildComplexityRouterConfig assistant turns", () => {
  const llmParams: BuildComplexityRouterConfigParams = {
    ...baseParams,
    classifierType: "llm",
    classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
  };

  it("emits the field when the LLM classifier is selected", () => {
    const config = buildComplexityRouterConfig({ ...llmParams, classifierContextIncludeAssistantTurns: true });
    expect(config.classifier_context_include_assistant_turns).toBe(true);
  });

  it("emits the switch turned off, since false is a choice the operator made and not an absent value", () => {
    const config = buildComplexityRouterConfig({ ...llmParams, classifierContextIncludeAssistantTurns: false });
    expect(config.classifier_context_include_assistant_turns).toBe(false);
  });

  it("omits it when classifier_type is heuristic even if a value lingers in state", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      classifierType: "heuristic",
      classifierContextIncludeAssistantTurns: true,
    });
    expect(config.classifier_context_include_assistant_turns).toBeUndefined();
  });

  it("omits it when unset, leaving the backend default", () => {
    const config = buildComplexityRouterConfig(llmParams);
    expect(config.classifier_context_include_assistant_turns).toBeUndefined();
  });
});

describe("classifier prompt and fallback", () => {
  const llmParams: BuildComplexityRouterConfigParams = {
    ...baseParams,
    classifierType: "llm",
    classifierLlmConfig: { model: "haiku-classifier", timeout_ms: 400 },
  };

  it("omits system_prompt when the operator never edited the prompt", () => {
    // The backend rejects a blank string, and storing a copy of the default would freeze the
    // rubric so later improvements never reach this router.
    const config = buildComplexityRouterConfig({
      ...llmParams,
      classifierLlmConfig: { model: "haiku-classifier", timeout_ms: 400, system_prompt: "   " },
    });
    expect(config.classifier_llm_config).toEqual({ model: "haiku-classifier", timeout_ms: 400 });
    expect(config.classifier_llm_config).not.toHaveProperty("system_prompt");
  });

  it("keeps a custom system_prompt verbatim, whitespace and all", () => {
    const systemPrompt = "  Grade data sensitivity.\n\nSIMPLE=public  ";
    const config = buildComplexityRouterConfig({
      ...llmParams,
      classifierLlmConfig: { model: "haiku-classifier", timeout_ms: 400, system_prompt: systemPrompt },
    });
    expect(config.classifier_llm_config?.system_prompt).toBe(systemPrompt);
  });

  it("emits classifier_fallback only for the llm classifier", () => {
    expect(buildComplexityRouterConfig({ ...llmParams, classifierFallback: "default_model" }).classifier_fallback).toBe(
      "default_model",
    );
    expect(buildComplexityRouterConfig({ ...baseParams, classifierFallback: "default_model" })).not.toHaveProperty(
      "classifier_fallback",
    );
  });

  it("omits classifier_fallback when unset so the backend default applies", () => {
    expect(buildComplexityRouterConfig(llmParams)).not.toHaveProperty("classifier_fallback");
  });

  it("sends the chat preset the operator picked", () => {
    const config = buildComplexityRouterConfig({
      ...llmParams,
      classifierLlmConfig: { model: "haiku-classifier", timeout_ms: 400, classification_rubric: "chat" },
    });
    expect(config.classifier_llm_config).toEqual({
      model: "haiku-classifier",
      timeout_ms: 400,
      classification_rubric: "chat",
    });
  });

  it("omits the preset when none is set, leaving an existing router on the rubric it already had", () => {
    // An unset preset means the pre-calibration rubric on the backend. Materializing a value here
    // would change the tier decisions, and the bill, of a router the operator only opened to edit.
    const config = buildComplexityRouterConfig(llmParams);
    expect(config.classifier_llm_config).not.toHaveProperty("classification_rubric");
  });

  it("drops the preset when a custom prompt replaces the rubric, which the backend rejects together", () => {
    const config = buildComplexityRouterConfig({
      ...llmParams,
      classifierLlmConfig: {
        model: "haiku-classifier",
        timeout_ms: 400,
        classification_rubric: "chat",
        system_prompt: "Grade the data sensitivity of the request.",
      },
    });
    expect(config.classifier_llm_config).not.toHaveProperty("classification_rubric");
    expect(config.classifier_llm_config?.system_prompt).toBe("Grade the data sensitivity of the request.");
  });

  it("normalizeClassifierLlmConfig leaves a real prompt untouched and strips an empty one", () => {
    expect(normalizeClassifierLlmConfig({ model: "m", timeout_ms: 1, system_prompt: "x" })).toEqual({
      model: "m",
      timeout_ms: 1,
      system_prompt: "x",
    });
    expect(normalizeClassifierLlmConfig({ model: "m", timeout_ms: 1, system_prompt: "" })).toEqual({
      model: "m",
      timeout_ms: 1,
    });
  });
});

describe("tier labels", () => {
  it("omits tier_labels entirely when the operator renamed nothing", () => {
    expect(buildComplexityRouterConfig(baseParams).tier_labels).toBeUndefined();
  });

  it("omits a label that only restates the default, so a later default change still reaches this router", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      tierLabels: { SIMPLE: "Simple", MEDIUM: "Medium", COMPLEX: "Complex", REASONING: "Reasoning" },
    });
    expect(config.tier_labels).toBeUndefined();
  });

  it("emits only the renamed tiers, trimmed, and leaves the tier keys canonical", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      tierLabels: { SIMPLE: "  Cheap  ", REASONING: "Deep" },
    });
    expect(config.tier_labels).toEqual({ SIMPLE: "Cheap", REASONING: "Deep" });
    expect(Object.keys(config.tiers)).toEqual(["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"]);
  });

  it("treats a whitespace-only label as no rename rather than sending a blank the backend rejects", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, tierLabels: { SIMPLE: "   " } });
    expect(config.tier_labels).toBeUndefined();
  });
});

describe("getTierLabelsError", () => {
  it("accepts an unrenamed router", () => {
    expect(getTierLabelsError(undefined)).toBeNull();
  });

  it("accepts a full distinct rename", () => {
    const fullRename = { SIMPLE: "Cheap", MEDIUM: "Standard", COMPLEX: "Premium", REASONING: "Deep" };
    expect(getTierLabelsError(fullRename)).toBeNull();
  });

  it("rejects two tiers sharing a name, which would be ambiguous in the logs", () => {
    expect(getTierLabelsError({ SIMPLE: "Cheap", MEDIUM: "Cheap" })).toMatch(/unique/i);
  });

  it("rejects names that differ only by case, since the logs would not tell them apart", () => {
    expect(getTierLabelsError({ SIMPLE: "Cheap", MEDIUM: "cheap" })).toMatch(/unique/i);
  });

  it("rejects a rename that collides with an untouched tier's name", () => {
    expect(getTierLabelsError({ SIMPLE: "Medium" })).toMatch(/another tier's name/i);
  });

  it("rejects a label that is another tier's canonical name", () => {
    expect(getTierLabelsError({ SIMPLE: "COMPLEX" })).toMatch(/another tier's name/i);
  });

  it("allows a label equal to that tier's own canonical name, which is a no-op", () => {
    expect(getTierLabelsError({ SIMPLE: "SIMPLE" })).toBeNull();
  });
});

describe("hydrateTierLabels", () => {
  it("returns undefined for a config that never set labels", () => {
    expect(hydrateTierLabels(undefined)).toBeUndefined();
  });

  it("keeps the stored labels", () => {
    expect(hydrateTierLabels({ SIMPLE: "Cheap", REASONING: "Deep" })).toEqual({ SIMPLE: "Cheap", REASONING: "Deep" });
  });

  it("drops non-string and blank values a hand-edited config could hold", () => {
    const handEdited = { SIMPLE: 7, MEDIUM: "  ", COMPLEX: null, REASONING: "Deep" };
    expect(hydrateTierLabels(handEdited)).toEqual({ REASONING: "Deep" });
  });

  it("ignores keys that are not tiers", () => {
    expect(hydrateTierLabels({ CHEAP: "Cheap" })).toBeUndefined();
  });

  it("returns undefined for a value that is not an object", () => {
    expect(hydrateTierLabels("Cheap")).toBeUndefined();
    expect(hydrateTierLabels(["Cheap"])).toBeUndefined();
  });
});

describe("buildComplexityRouterConfig scorer knobs", () => {
  const BOUNDARIES = { simple_medium: 0.22, medium_complex: 0.44, complex_reasoning: 0.66 };
  const tuned: BuildComplexityRouterConfigParams = { ...baseParams, tierBoundaries: BOUNDARIES };
  const llmWithDefaultFallback: BuildComplexityRouterConfigParams = {
    ...tuned,
    classifierType: "llm",
    classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
    classifierFallback: "default_model",
  };

  it("omits untouched knobs so the router tracks the backend defaults", () => {
    const config = buildComplexityRouterConfig(baseParams);

    expect(config).not.toHaveProperty("tier_boundaries");
    expect(config).not.toHaveProperty("dimension_weights");
  });

  it("emits what was set", () => {
    expect(buildComplexityRouterConfig(tuned).tier_boundaries).toEqual(BOUNDARIES);
  });

  it("drops them when the classifier falls back to the default model and nothing is scored", () => {
    expect(buildComplexityRouterConfig(llmWithDefaultFallback)).not.toHaveProperty("tier_boundaries");
  });

  it("omits the reasoning override floor while untouched, so it keeps tracking simple_medium", () => {
    expect(buildComplexityRouterConfig(baseParams)).not.toHaveProperty("reasoning_override_min_score");
  });

  it("emits the reasoning override floor that was set", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, reasoningOverrideMinScore: 0.4 });
    expect(config.reasoning_override_min_score).toBe(0.4);
  });

  // 0 is an unconditional override, not an absent knob, so a falsy check here would silently discard it.
  it("emits an explicit 0 reasoning override floor", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, reasoningOverrideMinScore: 0 });
    expect(config.reasoning_override_min_score).toBe(0);
  });

  it("drops the reasoning override floor when nothing is scored", () => {
    expect(buildComplexityRouterConfig({ ...llmWithDefaultFallback, reasoningOverrideMinScore: 0 })).not.toHaveProperty(
      "reasoning_override_min_score",
    );
  });
});

describe("plan-mode minimum tier", () => {
  it("omits plan_mode_min_tier when unset, so the backend default (off) is preserved", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, planModeMinTier: undefined });
    expect(config).not.toHaveProperty("plan_mode_min_tier");
  });

  it("writes the selected tier", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, planModeMinTier: "COMPLEX" });
    expect(config.plan_mode_min_tier).toBe("COMPLEX");
  });

  it("never writes an empty string, which the backend rejects instead of treating as off", () => {
    const config = buildComplexityRouterConfig({ ...baseParams, planModeMinTier: "  " });
    expect(config).not.toHaveProperty("plan_mode_min_tier");
  });
});

describe("getPlanModeTierError", () => {
  const tiersWithEmptyComplex = { SIMPLE: ["m1"], MEDIUM: ["m1"], COMPLEX: [], REASONING: [] };

  it("passes when the override is off", () => {
    expect(getPlanModeTierError(undefined, activeTierRows({ tiers: tiersWithEmptyComplex }))).toBeNull();
  });

  it("passes when the named tier has models", () => {
    expect(getPlanModeTierError("MEDIUM", activeTierRows({ tiers: tiersWithEmptyComplex }))).toBeNull();
  });

  it("blocks a tier whose models were removed, which the backend would reject with a 400", () => {
    expect(getPlanModeTierError("COMPLEX", activeTierRows({ tiers: tiersWithEmptyComplex }))).toContain("COMPLEX");
  });
});

describe("buildComplexityRouterConfig tier model params", () => {
  it("keeps tier_model_configs out of the payload when nothing is set", () => {
    expect(buildComplexityRouterConfig(baseParams)).not.toHaveProperty("tier_model_configs");
  });

  it("emits tier_model_configs beside string tiers when efforts are set", () => {
    const config = buildComplexityRouterConfig({
      ...baseParams,
      tierModelParams: { COMPLEX: { "claude-sonnet-4": { reasoning_effort: "high" } } },
    });
    expect(config.tiers).toEqual(tiers);
    expect(config.tier_model_configs).toEqual({
      COMPLEX: [{ model_name: "claude-sonnet-4", litellm_params: { reasoning_effort: "high" } }],
    });
  });
});

describe("getClassifierModelError", () => {
  it("stays quiet for a heuristic router, which needs no classifier model", () => {
    expect(getClassifierModelError({ classifier_type: "heuristic" })).toBeNull();
  });

  it("blocks an LLM classifier with no model, which the router cannot start without", () => {
    expect(getClassifierModelError({ classifier_type: "llm" })).toBe(
      "Please select a classifier model, or switch back to Heuristic",
    );
  });

  it("stays quiet once a model is chosen", () => {
    expect(
      getClassifierModelError({ classifier_type: "llm", classifier_llm_config: { model: "m", timeout_ms: 3000 } }),
    ).toBeNull();
  });
});

describe("getKeywordTierRulesError orphaned tiers", () => {
  const rows = activeTierRows({ tiers });

  it("accepts a rule naming a tier the router has", () => {
    expect(getKeywordTierRulesError([{ id: "r1", keywords: ["k"], tier: "COMPLEX" }], rows)).toBeNull();
  });

  it("names the rule pointing at a tier this router does not have", () => {
    expect(getKeywordTierRulesError([{ id: "r1", keywords: ["k"], tier: "AUDIT" }], rows)).toBe(
      "Keyword rule(s) 1 route to a tier this router no longer has",
    );
  });

  it("rejects a differently cased tier, because _validate_keyword_rule_tiers matches exactly", () => {
    expect(getKeywordTierRulesError([{ id: "r1", keywords: ["k"], tier: "complex" }], rows)).toBe(
      "Keyword rule(s) 1 route to a tier this router no longer has",
    );
  });

  it("reports an empty keyword row before an orphaned tier, since that is the nearer problem", () => {
    expect(getKeywordTierRulesError([{ id: "r1", keywords: [], tier: "AUDIT" }], rows)).toContain(
      "Add at least one keyword",
    );
  });
});

describe("heuristic_first", () => {
  const heuristicFirstParams: BuildComplexityRouterConfigParams = {
    ...baseParams,
    classifierType: "heuristic_first",
    heuristicFirstMaxTier: "SIMPLE",
    classifierLlmConfig: { model: "gpt-4o-mini", timeout_ms: 3000 },
    classifierContextWindowSize: 5,
    classifierContextBudgetChars: 4000,
    classifierFallback: "default_model",
  };

  it("emits heuristic_first_max_tier", () => {
    const config = buildComplexityRouterConfig(heuristicFirstParams);
    expect(config.classifier_type).toBe("heuristic_first");
    expect(config.heuristic_first_max_tier).toBe("SIMPLE");
  });

  it("keeps every classifier key the operator set, since heuristic_first still calls the classifier", () => {
    const config = buildComplexityRouterConfig(heuristicFirstParams);
    expect(config.classifier_llm_config).toEqual({ model: "gpt-4o-mini", timeout_ms: 3000 });
    expect(config.classifier_context_window_size).toBe(5);
    expect(config.classifier_context_budget_chars).toBe(4000);
    expect(config.classifier_fallback).toBe("default_model");
  });

  it("omits heuristic_first_max_tier on every other classifier type, which the backend rejects it on", () => {
    for (const classifierType of ["heuristic", "llm"] as const) {
      const config = buildComplexityRouterConfig({ ...heuristicFirstParams, classifierType });
      expect(config.heuristic_first_max_tier).toBeUndefined();
    }
  });
});
