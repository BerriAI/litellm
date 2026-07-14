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
};

const storedConfig = JSON.stringify(storedConfigValue);

const tiers = {
  SIMPLE: "gpt-4o-mini",
  MEDIUM: "gpt-4o-mini",
  COMPLEX: "anthropic-sonnet-4-5",
  REASONING: "anthropic-sonnet-4-5",
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
});
