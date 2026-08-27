import { KeywordTierRule } from "./KeywordTierRules";
import { type TierRow, activeTierName, tierRowById } from "./tier_rows";
import { emptyKeywordTierRuleIndexes, serializeKeywordTierRules } from "./complexity_router_keywords";
import { TierModelParams, TierModelParamsByTier, serializeTierModelConfigs } from "./complexity_router_tiers";
import {
  AdaptiveEligible,
  AdaptiveRouterWeights,
  ClassifierFallback,
  ClassifierLLMConfig,
  ClassifierType,
  ComplexityTierLabels,
  ComplexityRouterConfigValue,
  ComplexityTiers,
  DimensionWeights,
  TIER_KEYS,
  TIER_DESCRIPTIONS,
  TierBoundaries,
  TokenThresholds,
  effectiveTierLabel,
  heuristicScoringRoleFor,
  usesLlmClassifier,
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

interface ScorerKnobInputs {
  classifierType: ClassifierType;
  classifierFallback: ClassifierFallback | undefined;
  tierBoundaries: TierBoundaries | undefined;
  tokenThresholds: TokenThresholds | undefined;
  dimensionWeights: DimensionWeights | undefined;
  reasoningOverrideMinScore: number | undefined;
}

/**
 * The scorer knobs to persist, which is none of them on a router that never scores: an LLM classifier
 * falling back to the default model would otherwise carry settings that can only mislead the next reader.
 * Each is omitted while untouched, so the router keeps tracking the backend defaults.
 */
const scorerKnobPayload = ({
  classifierType,
  classifierFallback,
  tierBoundaries,
  tokenThresholds,
  dimensionWeights,
  reasoningOverrideMinScore,
}: ScorerKnobInputs) =>
  heuristicScoringRoleFor(classifierType, classifierFallback) === "never"
    ? {}
    : {
        ...(tierBoundaries && { tier_boundaries: tierBoundaries }),
        ...(tokenThresholds && { token_thresholds: tokenThresholds }),
        ...(dimensionWeights && { dimension_weights: dimensionWeights }),
        ...(reasoningOverrideMinScore !== undefined && { reasoning_override_min_score: reasoningOverrideMinScore }),
      };

export interface BuildComplexityRouterConfigParams {
  tiers: ComplexityTiers;
  defaultModel: string | undefined;
  planModeMinTier: string | undefined;
  tierLabels: ComplexityTierLabels | undefined;
  classifierType: ClassifierType;
  classifierLlmConfig: ClassifierLLMConfig | undefined;
  classifierContextWindowSize: number | undefined;
  classifierContextBudgetChars: number | undefined;
  classifierContextIncludeAssistantTurns: boolean | undefined;
  classifierFallback: ClassifierFallback | undefined;
  heuristicFirstMaxTier: string | undefined;
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
  tierBoundaries?: TierBoundaries;
  tokenThresholds?: TokenThresholds;
  dimensionWeights?: DimensionWeights;
  reasoningOverrideMinScore?: number;
  tierModelParams?: TierModelParamsByTier;
}

export interface ComplexityRouterConfigPayload {
  tiers: ComplexityTiers;
  default_model?: string;
  plan_mode_min_tier?: string;
  tier_labels?: ComplexityTierLabels;
  classifier_type: ClassifierType;
  classifier_llm_config?: ClassifierLLMConfig;
  classifier_context_window_size?: number;
  classifier_context_budget_chars?: number;
  classifier_context_per_turn_chars?: number;
  classifier_context_include_assistant_turns?: boolean;
  classifier_fallback?: ClassifierFallback;
  heuristic_first_max_tier?: string;
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
  tier_boundaries?: TierBoundaries;
  token_thresholds?: TokenThresholds;
  dimension_weights?: DimensionWeights;
  reasoning_override_min_score?: number;
  tier_model_configs?: Record<string, { model_name: string; litellm_params: TierModelParams }[]>;
}

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

// Requires every active tier non-empty, so the create form can never reach the
// resolveComplexityDefaultModel === undefined case. The edit modal allows a partially filled
// set, which is why it keeps its own !defaultModel guard after deriving.
export const getMissingTiersError = (rows: readonly TierRow[]): string | null => {
  const missing = rows.filter((row) => row.models.length === 0).map(activeTierName);
  if (missing.length === 0) return null;
  return `Select a model for the following tier(s): ${missing.join(", ")}`;
};

export const getPlanModeTierError = (planModeMinTier: string | undefined, rows: readonly TierRow[]): string | null => {
  if (!planModeMinTier) return null;
  const floor = tierRowById(rows, planModeMinTier);
  if (floor && floor.models.length > 0) return null;
  return `The plan-mode minimum tier (${floor ? activeTierName(floor) : planModeMinTier}) has no models. Add one or turn the override off.`;
};

// The tier is a free string since #37413, and _validate_keyword_rule_tiers matches it EXACTLY, so a
// rule naming a tier this router does not have is a raw 400 unless the gate catches it first.
export const getKeywordTierRulesError = (
  keywordTierRules: KeywordTierRule[],
  rows: readonly TierRow[],
): string | null => {
  const emptyRows = emptyKeywordTierRuleIndexes(keywordTierRules);
  if (emptyRows.length > 0)
    return `Add at least one keyword to keyword rule(s): ${emptyRows.map((index) => index + 1).join(", ")}`;
  const names = rows.map(activeTierName);
  const orphaned = keywordTierRules.flatMap((rule, index) => (names.includes(rule.tier) ? [] : [index + 1]));
  if (orphaned.length === 0) return null;
  return `Keyword rule(s) ${orphaned.join(", ")} route to a tier this router no longer has`;
};

// The submit gate and the submit handler both read this, so a disabled button and a refused submit
// cannot disagree about why.
export const getClassifierModelError = (
  config: Pick<ComplexityRouterConfigValue, "classifier_type" | "classifier_llm_config">,
): string | null =>
  usesLlmClassifier(config.classifier_type) && !config.classifier_llm_config?.model
    ? "Please select a classifier model, or switch back to Heuristic"
    : null;

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
  defaultModel,
  planModeMinTier,
  tierLabels,
  classifierType,
  classifierLlmConfig,
  classifierContextWindowSize,
  classifierContextBudgetChars,
  classifierContextIncludeAssistantTurns,
  classifierFallback,
  heuristicFirstMaxTier,
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
  tierBoundaries,
  tokenThresholds,
  dimensionWeights,
  reasoningOverrideMinScore,
  tierModelParams,
}: BuildComplexityRouterConfigParams): ComplexityRouterConfigPayload => {
  const serializedTierModelConfigs = serializeTierModelConfigs(tiers, tierModelParams);
  const cleanedEscalationKeywords = escalationKeywords.map((keyword) => keyword.trim()).filter(Boolean);
  const cleanedKeywordTierRules = serializeKeywordTierRules(keywordTierRules);
  const cleanedTierLabels = serializeTierLabels(tierLabels);
  const scorerInputs = {
    classifierType,
    classifierFallback,
    tierBoundaries,
    tokenThresholds,
    dimensionWeights,
    reasoningOverrideMinScore,
  };
  const scorerKnobs = scorerKnobPayload(scorerInputs);

  return {
    tiers,
    ...(serializedTierModelConfigs && { tier_model_configs: serializedTierModelConfigs }),
    ...(defaultModel?.trim() && { default_model: defaultModel }),
    ...(planModeMinTier?.trim() && { plan_mode_min_tier: planModeMinTier }),
    ...(cleanedTierLabels && { tier_labels: cleanedTierLabels }),
    classifier_type: classifierType,
    ...(usesLlmClassifier(classifierType) &&
      classifierLlmConfig && { classifier_llm_config: normalizeClassifierLlmConfig(classifierLlmConfig) }),
    ...(usesLlmClassifier(classifierType) &&
      classifierFallback !== undefined && { classifier_fallback: classifierFallback }),
    ...(classifierType === "heuristic_first" &&
      heuristicFirstMaxTier?.trim() && { heuristic_first_max_tier: heuristicFirstMaxTier }),
    ...(usesLlmClassifier(classifierType) &&
      classifierContextWindowSize !== undefined && {
        classifier_context_window_size: classifierContextWindowSize,
      }),
    ...(usesLlmClassifier(classifierType) &&
      classifierContextBudgetChars !== undefined && {
        classifier_context_budget_chars: classifierContextBudgetChars,
      }),
    ...(usesLlmClassifier(classifierType) &&
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
    ...scorerKnobs,
  };
};
