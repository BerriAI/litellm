import { buildUpdatedComplexityRouterConfig } from "./edit_auto_router_modal";

const storedConfigValue = {
  tiers: {
    SIMPLE: "old-simple",
    MEDIUM: "old-medium",
    COMPLEX: "old-complex",
    REASONING: "old-reasoning",
  },
  classifier_type: "llm",
  classifier_llm_config: { model: "old-classifier", timeout_ms: 1200 },
  custom_technical_keywords: ["kafka", "terraform"],
  keyword_tier_rules: [{ keywords: ["invoice", "refund"], tier: "MEDIUM" }],
  semantic_keyword_matching: true,
  embedding_model: "voyage-4-large",
  match_threshold: 0.65,
  adaptive: true,
  adaptive_weights: { quality: 0.3, cost: 0.7 },
  tier_distance_penalty: 0.8,
  adaptive_eligible: "all",
  return_raw_model_name: true,
};

const storedConfig = JSON.stringify(storedConfigValue);

const tiers = {
  SIMPLE: ["gpt-4o-mini"],
  MEDIUM: ["gpt-4o-mini"],
  COMPLEX: ["anthropic-sonnet-4-5"],
  REASONING: ["anthropic-sonnet-4-5"],
};

const classifiedTierValue = {
  tiers,
  classifier_type: "heuristic" as const,
  adaptive: true,
  adaptive_weights: { quality: 0.4, cost: 0.6 },
  tier_distance_penalty: 0.8,
  adaptive_eligible: "classified_tier" as const,
};

const expectedClassifiedTierConfig = {
  tiers,
  classifier_type: "heuristic",
  custom_technical_keywords: ["kafka", "terraform"],
  keyword_tier_rules: [{ keywords: ["invoice", "refund"], tier: "MEDIUM" }],
  semantic_keyword_matching: true,
  embedding_model: "voyage-4-large",
  match_threshold: 0.65,
  session_affinity: false,
  deployment_affinity: true,
  adaptive: true,
  adaptive_weights: { quality: 0.4, cost: 0.6 },
  adaptive_eligible: "classified_tier",
};

const adaptiveDisabledValue = {
  tiers,
  classifier_type: "heuristic" as const,
  adaptive: false,
};

const expectedAdaptiveDisabledConfig = {
  tiers,
  classifier_type: "heuristic",
  custom_technical_keywords: ["kafka", "terraform"],
  keyword_tier_rules: [{ keywords: ["invoice", "refund"], tier: "MEDIUM" }],
  semantic_keyword_matching: true,
  embedding_model: "voyage-4-large",
  match_threshold: 0.65,
  session_affinity: false,
  deployment_affinity: true,
};

describe("buildUpdatedComplexityRouterConfig", () => {
  it("preserves unrelated options and omits the penalty for classified-tier routing", () => {
    const updatedConfig = buildUpdatedComplexityRouterConfig(storedConfig, classifiedTierValue);

    expect(updatedConfig).toEqual(expectedClassifiedTierConfig);
  });

  it("removes managed adaptive and classifier fields when they are disabled", () => {
    const updatedConfig = buildUpdatedComplexityRouterConfig(storedConfig, adaptiveDisabledValue);

    expect(updatedConfig).toEqual(expectedAdaptiveDisabledConfig);
  });

  it("includes return_raw_model_name only when enabled", () => {
    const updatedConfig = buildUpdatedComplexityRouterConfig(storedConfig, {
      ...classifiedTierValue,
      return_raw_model_name: true,
    });

    expect(updatedConfig.return_raw_model_name).toBe(true);
  });

  it("updates custom technical keywords when they are edited", () => {
    const updatedConfig = buildUpdatedComplexityRouterConfig(storedConfig, classifiedTierValue, ["postgres"]);

    expect(updatedConfig.custom_technical_keywords).toEqual(["postgres"]);
  });

  it("removes custom technical keywords when they are cleared", () => {
    const updatedConfig = buildUpdatedComplexityRouterConfig(storedConfig, classifiedTierValue, []);

    expect(updatedConfig.custom_technical_keywords).toBeUndefined();
  });

  it("writes the plugin and its timeout for a custom classifier", () => {
    const updatedConfig = buildUpdatedComplexityRouterConfig(storedConfig, {
      tiers,
      classifier_type: "custom" as const,
      classifier_plugin: "tier-by-team",
      classifier_plugin_timeout_ms: 1500,
      classifier_fallback: "default_model" as const,
    });

    expect(updatedConfig).toMatchObject({
      classifier_type: "custom",
      classifier_plugin: "tier-by-team",
      classifier_plugin_timeout_ms: 1500,
      classifier_fallback: "default_model",
    });
    expect(updatedConfig.classifier_llm_config).toBeUndefined();
  });

  // Both keys are managed, so switching off custom has to remove them: the backend rejects a stored
  // classifier_plugin sitting next to a non-custom classifier_type.
  it("removes a stored plugin and its timeout when the classifier is no longer custom", () => {
    const storedCustom = JSON.stringify({
      ...storedConfigValue,
      classifier_type: "custom",
      classifier_plugin: "tier-by-team",
      classifier_plugin_timeout_ms: 1500,
    });

    const updatedConfig = buildUpdatedComplexityRouterConfig(storedCustom, classifiedTierValue);

    expect(updatedConfig.classifier_plugin).toBeUndefined();
    expect(updatedConfig.classifier_plugin_timeout_ms).toBeUndefined();
  });

  it("preserves a tier configured with more than one model as a pool", () => {
    const multiModelValue = {
      ...classifiedTierValue,
      tiers: { ...tiers, SIMPLE: ["gpt-4o-mini", "claude-haiku-4-5"] },
    };
    const updatedConfig = buildUpdatedComplexityRouterConfig(storedConfig, multiModelValue);

    expect(updatedConfig.tiers).toMatchObject({ SIMPLE: ["gpt-4o-mini", "claude-haiku-4-5"] });
  });
});
