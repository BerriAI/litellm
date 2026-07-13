import { KeywordTierRule } from "./KeywordTierRules";
import { ClassifierLLMConfig, ClassifierType } from "./ComplexityRouterConfig";

export interface ComplexityTiers {
  SIMPLE: string;
  MEDIUM: string;
  COMPLEX: string;
  REASONING: string;
}

export interface BuildComplexityRouterConfigParams {
  tiers: ComplexityTiers;
  classifierType: ClassifierType;
  classifierLlmConfig: ClassifierLLMConfig | undefined;
  customTechnicalKeywords: string[];
  keywordTierRules: KeywordTierRule[];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
  matchThreshold: number;
}

export interface ComplexityRouterConfigPayload {
  tiers: ComplexityTiers;
  classifier_type: ClassifierType;
  classifier_llm_config?: ClassifierLLMConfig;
  custom_technical_keywords?: string[];
  keyword_tier_rules?: { keywords: string[]; tier: KeywordTierRule["tier"] }[];
  semantic_keyword_matching?: boolean;
  embedding_model?: string;
  match_threshold?: number;
}

const TIER_KEYS: Array<keyof ComplexityTiers> = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"];

export const getMissingTiersError = (tiers: ComplexityTiers): string | null => {
  const missing = TIER_KEYS.filter((tier) => !tiers[tier]);
  if (missing.length === 0) return null;
  return `Select a model for the following tier(s): ${missing.join(", ")}`;
};

export const getSemanticConfigError = ({
  semanticMatchingEnabled,
  embeddingModel,
  keywordTierRules,
}: Pick<BuildComplexityRouterConfigParams, "semanticMatchingEnabled" | "embeddingModel" | "keywordTierRules">):
  string | null => {
  if (!semanticMatchingEnabled) return null;
  if (!embeddingModel) return "Select an embedding model to use semantic keyword matching";
  if (keywordTierRules.length === 0) return "Add at least one keyword tier rule to use semantic keyword matching";
  if (keywordTierRules.some((rule) => !rule.keywords.some((keyword) => keyword.trim())))
    return "Every keyword tier rule needs at least one keyword";
  return null;
};

export const buildComplexityRouterConfig = ({
  tiers,
  classifierType,
  classifierLlmConfig,
  customTechnicalKeywords,
  keywordTierRules,
  semanticMatchingEnabled,
  embeddingModel,
  matchThreshold,
}: BuildComplexityRouterConfigParams): ComplexityRouterConfigPayload => {
  // Trim keywords and drop empty ones; drop any rule left with no keywords. Clicking
  // "Add keyword rule" seeds a rule with an empty keywords list, so without this an
  // unfilled row (common in the heuristic flow, where getSemanticConfigError doesn't run)
  // would ship keyword_tier_rules the backend validator rejects with a 400.
  const cleanedKeywordTierRules = keywordTierRules
    .map((rule) => ({ keywords: rule.keywords.map((k) => k.trim()).filter(Boolean), tier: rule.tier }))
    .filter((rule) => rule.keywords.length > 0);

  return {
    tiers,
    classifier_type: classifierType,
    ...(classifierType === "llm" && classifierLlmConfig && { classifier_llm_config: classifierLlmConfig }),
    ...(customTechnicalKeywords.length > 0 && { custom_technical_keywords: customTechnicalKeywords }),
    ...(cleanedKeywordTierRules.length > 0 && { keyword_tier_rules: cleanedKeywordTierRules }),
    ...(semanticMatchingEnabled && {
      semantic_keyword_matching: true,
      embedding_model: embeddingModel,
      match_threshold: matchThreshold,
    }),
  };
};
