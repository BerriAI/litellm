import { KeywordTierRule } from "./KeywordTierRules";
import { emptyKeywordTierRuleIndexes, serializeKeywordTierRules } from "./complexity_router_keywords";
import {
  AdaptiveEligible,
  AdaptiveRouterWeights,
  ClassifierFallback,
  ClassifierLLMConfig,
  ClassifierType,
  ComplexityTierLabels,
  ComplexityTiers,
  TIER_DESCRIPTIONS,
  effectiveTierLabel,
} from "./ComplexityRouterConfig";

/**
 * Drop an empty system_prompt so the payload carries an override only when there is one. The
 * backend rejects a blank string rather than reading it as "use the default", and sending `""`
 * would turn an untouched editor into a validation error.
 *
 * A custom prompt is the classifier's whole system role, so the backend also rejects a rubric preset
 * sent alongside one. Each branch rebuilds the object rather than spreading it, so a preset left on
 * the form state from before the prompt was written cannot reach the wire and fail the save.
 *
 * An untouched picker sends no rubric at all rather than a copy of the default it displays. The
 * backend reads absence as "use the default preset", so omitting it keeps a router the operator never
 * configured on whatever that default becomes, and keeps routers built here behaving the same as ones
 * written by hand in config.
 */
export const normalizeClassifierLlmConfig = ({
  model,
  timeout_ms,
  classification_rubric,
  system_prompt,
}: ClassifierLLMConfig): ClassifierLLMConfig =>
  system_prompt?.trim()
    ? { model, timeout_ms, system_prompt }
    : { model, timeout_ms, ...(classification_rubric && { classification_rubric }) };

export interface BuildComplexityRouterConfigParams {
  tiers: ComplexityTiers;
  tierLabels: ComplexityTierLabels | undefined;
  classifierType: ClassifierType;
  classifierLlmConfig: ClassifierLLMConfig | undefined;
  classifierContextWindowSize: number | undefined;
  classifierContextPerTurnChars: number | undefined;
  classifierContextIncludeAssistantTurns: boolean | undefined;
  classifierFallback: ClassifierFallback | undefined;
  sessionAffinity: boolean;
  deploymentAffinity: boolean;
  customTechnicalKeywords: string[];
  keywordTierRules: KeywordTierRule[];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
  matchThreshold: number;
  escalationKeywords: string[];
  adaptive: boolean;
  adaptiveWeights: AdaptiveRouterWeights;
  tierDistancePenalty: number;
  adaptiveEligible: AdaptiveEligible;
  returnRawModelName: boolean;
}

export interface ComplexityRouterConfigPayload {
  tiers: ComplexityTiers;
  tier_labels?: ComplexityTierLabels;
  classifier_type: ClassifierType;
  classifier_llm_config?: ClassifierLLMConfig;
  classifier_context_window_size?: number;
  classifier_context_per_turn_chars?: number;
  classifier_context_include_assistant_turns?: boolean;
  classifier_fallback?: ClassifierFallback;
  session_affinity: boolean;
  deployment_affinity: boolean;
  custom_technical_keywords?: string[];
  keyword_tier_rules?: { keywords: string[]; tier: KeywordTierRule["tier"] }[];
  semantic_keyword_matching?: boolean;
  embedding_model?: string;
  match_threshold?: number;
  escalation_keywords?: string[];
  adaptive?: boolean;
  adaptive_weights?: AdaptiveRouterWeights;
  tier_distance_penalty?: number;
  adaptive_eligible?: AdaptiveEligible;
  return_raw_model_name?: boolean;
}

const TIER_KEYS: Array<keyof ComplexityTiers> = ["SIMPLE", "MEDIUM", "COMPLEX", "REASONING"];

export const serializeTierLabels = (tierLabels: ComplexityTierLabels | undefined): ComplexityTierLabels | undefined => {
  const renamed = TIER_KEYS.map((tier) => [tier, tierLabels?.[tier]?.trim() ?? ""] as const).filter(
    ([tier, label]) => label !== "" && label !== TIER_DESCRIPTIONS[tier].label,
  );
  if (renamed.length === 0) return undefined;
  return Object.fromEntries(renamed);
};

export const hydrateTierLabels = (stored: unknown): ComplexityTierLabels | undefined => {
  if (typeof stored !== "object" || stored === null || Array.isArray(stored)) return undefined;
  const entries = TIER_KEYS.map((tier) => [tier, (stored as Record<string, unknown>)[tier]] as const).filter(
    (entry): entry is readonly [keyof ComplexityTiers, string] =>
      typeof entry[1] === "string" && entry[1].trim() !== "",
  );
  if (entries.length === 0) return undefined;
  return Object.fromEntries(entries);
};

export const getTierLabelsError = (tierLabels: ComplexityTierLabels | undefined): string | null => {
  const shadowing = TIER_KEYS.filter((tier) => {
    const label = tierLabels?.[tier]?.trim().toUpperCase() ?? "";
    return label !== "" && label !== tier && (TIER_KEYS as string[]).includes(label);
  });
  if (shadowing.length > 0) {
    return `A tier's display name can't be another tier's name: ${shadowing.join(", ")}`;
  }
  const labels = TIER_KEYS.map((tier) => effectiveTierLabel(tier, tierLabels).toLowerCase());
  const duplicates = Array.from(new Set(labels.filter((label, index) => labels.indexOf(label) !== index)));
  if (duplicates.length > 0) {
    return `Tier display names must be unique. Repeated: ${duplicates.join(", ")}`;
  }
  return null;
};

export const getMissingTiersError = (tiers: ComplexityTiers): string | null => {
  const missing = TIER_KEYS.filter((tier) => tiers[tier].length === 0);
  if (missing.length === 0) return null;
  return `Select a model for the following tier(s): ${missing.join(", ")}`;
};

export const getKeywordTierRulesError = (keywordTierRules: KeywordTierRule[]): string | null => {
  const emptyRows = emptyKeywordTierRuleIndexes(keywordTierRules);
  if (emptyRows.length === 0) return null;
  return `Add at least one keyword to keyword rule(s): ${emptyRows.map((index) => index + 1).join(", ")}`;
};

export const getSemanticConfigError = ({
  semanticMatchingEnabled,
  embeddingModel,
  keywordTierRules,
}: Pick<BuildComplexityRouterConfigParams, "semanticMatchingEnabled" | "embeddingModel" | "keywordTierRules">):
  | string
  | null => {
  if (!semanticMatchingEnabled) return null;
  if (!embeddingModel) return "Select an embedding model to use semantic keyword matching";
  if (keywordTierRules.length === 0) return "Add at least one keyword tier rule to use semantic keyword matching";
  return null;
};

export const buildComplexityRouterConfig = ({
  tiers,
  tierLabels,
  classifierType,
  classifierLlmConfig,
  classifierContextWindowSize,
  classifierContextPerTurnChars,
  classifierContextIncludeAssistantTurns,
  classifierFallback,
  sessionAffinity,
  deploymentAffinity,
  customTechnicalKeywords,
  keywordTierRules,
  semanticMatchingEnabled,
  embeddingModel,
  matchThreshold,
  escalationKeywords,
  adaptive,
  adaptiveWeights,
  tierDistancePenalty,
  adaptiveEligible,
  returnRawModelName,
}: BuildComplexityRouterConfigParams): ComplexityRouterConfigPayload => {
  const cleanedEscalationKeywords = escalationKeywords.map((keyword) => keyword.trim()).filter(Boolean);
  const cleanedKeywordTierRules = serializeKeywordTierRules(keywordTierRules);
  const cleanedTierLabels = serializeTierLabels(tierLabels);

  return {
    tiers,
    ...(cleanedTierLabels && { tier_labels: cleanedTierLabels }),
    classifier_type: classifierType,
    ...(classifierType === "llm" &&
      classifierLlmConfig && { classifier_llm_config: normalizeClassifierLlmConfig(classifierLlmConfig) }),
    ...(classifierType === "llm" && classifierFallback !== undefined && { classifier_fallback: classifierFallback }),
    ...(classifierType === "llm" &&
      classifierContextWindowSize !== undefined && {
        classifier_context_window_size: classifierContextWindowSize,
      }),
    ...(classifierType === "llm" &&
      classifierContextPerTurnChars !== undefined && {
        classifier_context_per_turn_chars: classifierContextPerTurnChars,
      }),
    ...(classifierType === "llm" &&
      classifierContextIncludeAssistantTurns !== undefined && {
        classifier_context_include_assistant_turns: classifierContextIncludeAssistantTurns,
      }),
    session_affinity: sessionAffinity,
    deployment_affinity: deploymentAffinity,
    ...(customTechnicalKeywords.length > 0 && { custom_technical_keywords: customTechnicalKeywords }),
    ...(cleanedKeywordTierRules.length > 0 && { keyword_tier_rules: cleanedKeywordTierRules }),
    escalation_keywords: cleanedEscalationKeywords,
    ...(semanticMatchingEnabled && {
      semantic_keyword_matching: true,
      embedding_model: embeddingModel,
      match_threshold: matchThreshold,
    }),
    ...(adaptive && {
      adaptive: true,
      adaptive_weights: adaptiveWeights,
      ...(adaptiveEligible === "all" && { tier_distance_penalty: tierDistancePenalty }),
      adaptive_eligible: adaptiveEligible,
    }),
    ...(returnRawModelName && { return_raw_model_name: true }),
  };
};
