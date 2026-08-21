import { KeywordTierRule } from "./KeywordTierRules";
import { findTierByName, tierNamesMatch } from "./custom_tier_set";
import { emptyKeywordTierRuleIndexes, serializeKeywordTierRules } from "./complexity_router_keywords";
import {
  TierModelParams,
  TierModelParamsByTier,
  normalizeTierModels,
  serializeTierModelConfigs,
} from "./complexity_router_tiers";
import {
  AdaptiveEligible,
  AdaptiveRouterWeights,
  ClassifierFallback,
  ClassifierLLMConfig,
  ClassifierType,
  ComplexityTierLabels,
  ComplexityTiers,
  CustomTierSet,
  DimensionWeights,
  TIER_DESCRIPTIONS,
  TIER_KEYS,
  TierBoundaries,
  TokenThresholds,
  effectiveTierLabel,
  heuristicScoringRoleFor,
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
  customTierSet?: CustomTierSet;
  defaultModel: string | undefined;
  planModeMinTier: string | undefined;
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
  tierBoundaries?: TierBoundaries;
  tokenThresholds?: TokenThresholds;
  dimensionWeights?: DimensionWeights;
  reasoningOverrideMinScore?: number;
  tierModelParams?: TierModelParamsByTier;
}

export interface TierDefinitionPayload {
  name: string;
  description?: string;
}

export interface ComplexityRouterConfigPayload {
  tiers: ComplexityTiers | Record<string, string[]>;
  tier_definitions?: TierDefinitionPayload[];
  fallback_tier?: string;
  default_model?: string;
  plan_mode_min_tier?: string;
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

// Requires all 4 tiers non-empty, so the create form can never reach the
// resolveComplexityDefaultModel(tiers, ...) === undefined case — MEDIUM (or SIMPLE) is always
// populated. The edit modal has no equivalent of this check (it allows saving with only some
// tiers filled), which is why it needs its own explicit `!defaultModel` guard after deriving —
// see edit_auto_router_modal.tsx's save handler. A future contributor copying this form's submit
// handler elsewhere should not assume the same guarantee holds without this check.
export const getMissingTiersError = (tiers: ComplexityTiers): string | null => {
  const missing = TIER_KEYS.filter((tier) => tiers[tier].length === 0);
  if (missing.length === 0) return null;
  return `Select a model for the following tier(s): ${missing.join(", ")}`;
};

// The backend rejects a plan-mode floor naming a tier with no models. The create form's
// getMissingTiersError makes this unreachable there; the edit modal allows partially filled
// tiers, so both gates call this to keep the two forms symmetric.
export const getPlanModeTierError = (planModeMinTier: string | undefined, tiers: ComplexityTiers): string | null => {
  if (!planModeMinTier) return null;
  const models = tiers[planModeMinTier as keyof ComplexityTiers] ?? [];
  if (models.length > 0) return null;
  return `The plan-mode minimum tier (${planModeMinTier}) has no models. Add one or turn the override off.`;
};

// Rules point at a tier by NAME, so a tier edit can orphan one and the backend rejects the config.
export const getKeywordTierRulesError = (
  keywordTierRules: KeywordTierRule[],
  activeTierNames?: readonly string[],
): string | null => {
  const emptyRows = emptyKeywordTierRuleIndexes(keywordTierRules);
  if (emptyRows.length > 0)
    return `Add at least one keyword to keyword rule(s): ${emptyRows.map((index) => index + 1).join(", ")}`;
  if (!activeTierNames) return null;
  const orphaned = keywordTierRules.flatMap((rule, index) =>
    activeTierNames.some((name) => tierNamesMatch(name, rule.tier)) ? [] : [index + 1],
  );
  if (orphaned.length === 0) return null;
  return `Keyword rule(s) ${orphaned.join(", ")} route to a tier this router no longer has`;
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

// Mirrors validate_complexity_router_config_write (litellm/router_utils/auto_router_model_naming.py).
export const KEYS_REJECTED_WITH_CUSTOM_TIERS: readonly string[] = [
  "plugins",
  "tier_labels",
  "classifier_fallback",
  "adaptive",
  "adaptive_weights",
  "tier_distance_penalty",
  "adaptive_eligible",
  "tier_boundaries",
  "token_thresholds",
  "dimension_weights",
  "reasoning_override_min_score",
  "escalation_keywords",
  "session_affinity",
];

export const serializeCustomTierSet = (
  customTierSet: CustomTierSet,
): Pick<ComplexityRouterConfigPayload, "tiers" | "tier_definitions" | "fallback_tier"> => {
  const fallbackName = customTierSet.tiers.find((row) => row.id === customTierSet.fallback_tier_id)?.name.trim();
  return {
    tiers: Object.fromEntries(customTierSet.tiers.map((row) => [row.name.trim(), row.models] as const)),
    tier_definitions: customTierSet.tiers.map((row) => ({
      name: row.name.trim(),
      ...(row.definition.trim() && { description: row.definition.trim() }),
    })),
    ...(fallbackName && { fallback_tier: fallbackName }),
  };
};

// A built-in name keeps its canonical key as the row id so Restore recognizes it.
export const hydrateCustomTierSet = (parsedConfig: {
  tier_definitions?: unknown;
  fallback_tier?: unknown;
  tiers?: unknown;
}): CustomTierSet | undefined => {
  if (!Array.isArray(parsedConfig.tier_definitions) || parsedConfig.tier_definitions.length === 0) return undefined;
  const storedTiers =
    typeof parsedConfig.tiers === "object" && parsedConfig.tiers !== null && !Array.isArray(parsedConfig.tiers)
      ? (parsedConfig.tiers as Record<string, unknown>)
      : {};
  const rows = parsedConfig.tier_definitions.flatMap((entry, index): CustomTierSet["tiers"] => {
    if (typeof entry !== "object" || entry === null) return [];
    const { name, description } = entry as { name?: unknown; description?: unknown };
    if (typeof name !== "string" || !name.trim()) return [];
    return [
      {
        id: TIER_KEYS.find((tier) => tierNamesMatch(tier, name)) ?? `stored-${index}`,
        name: name.trim(),
        definition: typeof description === "string" ? description.trim() : "",
        models: normalizeTierModels(Object.entries(storedTiers).find(([tier]) => tierNamesMatch(tier, name))?.[1]),
      },
    ];
  });
  if (rows.length === 0) return undefined;
  const storedFallback = typeof parsedConfig.fallback_tier === "string" ? parsedConfig.fallback_tier.trim() : "";
  return { tiers: rows, fallback_tier_id: findTierByName(rows, storedFallback)?.id ?? "" };
};

// The floor is a ROW ID on the value; one rule at every layer: unresolvable means OFF.
export const hydratePlanModeMinTier = (
  stored: unknown,
  customTierSet: CustomTierSet | undefined,
): string | undefined => {
  if (typeof stored !== "string" || stored.trim() === "") return undefined;
  if (!customTierSet) return stored;
  return findTierByName(customTierSet.tiers, stored)?.id;
};

// Everything an edited tier set forces onto the wire; shared by both builders.
export const customTierSetWireFields = (
  customTierSet: CustomTierSet,
  classifierLlmConfig: ClassifierLLMConfig | undefined,
  planModeMinTierId: string | undefined,
) => {
  const planModeName = customTierSet.tiers.find((row) => row.id === planModeMinTierId)?.name.trim();
  return {
    ...serializeCustomTierSet(customTierSet),
    classifier_type: "llm" as const,
    ...(classifierLlmConfig && {
      classifier_llm_config: { model: classifierLlmConfig.model, timeout_ms: classifierLlmConfig.timeout_ms },
    }),
    session_affinity: false,
    escalation_keywords: [] as string[],
    ...(planModeName && { plan_mode_min_tier: planModeName }),
  };
};

export { getCustomTierRowsError } from "./custom_tier_set";

export const buildComplexityRouterConfig = ({
  tiers,
  customTierSet,
  defaultModel,
  planModeMinTier,
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
  tierBoundaries,
  tokenThresholds,
  dimensionWeights,
  reasoningOverrideMinScore,
  tierModelParams,
}: BuildComplexityRouterConfigParams): ComplexityRouterConfigPayload => {
  const serializedTierModelConfigs = customTierSet ? undefined : serializeTierModelConfigs(tiers, tierModelParams);
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

  // A custom set forces the LLM classifier, so llm-only inputs must survive a stale heuristic field.
  const effectiveType: ClassifierType = customTierSet ? "llm" : classifierType;
  const payload: ComplexityRouterConfigPayload = {
    tiers,
    ...(serializedTierModelConfigs && { tier_model_configs: serializedTierModelConfigs }),
    ...(defaultModel?.trim() && { default_model: defaultModel }),
    ...(planModeMinTier?.trim() && { plan_mode_min_tier: planModeMinTier }),
    ...(cleanedTierLabels && { tier_labels: cleanedTierLabels }),
    classifier_type: classifierType,
    ...(effectiveType === "llm" &&
      classifierLlmConfig && { classifier_llm_config: normalizeClassifierLlmConfig(classifierLlmConfig) }),
    ...(classifierType === "llm" && classifierFallback !== undefined && { classifier_fallback: classifierFallback }),
    ...(effectiveType === "llm" &&
      classifierContextWindowSize !== undefined && {
        classifier_context_window_size: classifierContextWindowSize,
      }),
    ...(effectiveType === "llm" &&
      classifierContextPerTurnChars !== undefined && {
        classifier_context_per_turn_chars: classifierContextPerTurnChars,
      }),
    ...(effectiveType === "llm" &&
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
  if (!customTierSet) return payload;
  // Drop what the backend rejects beside tier_definitions; stale control state would fail the save.
  const { plan_mode_min_tier: planModeMinTierId, ...withFloorRemoved } = payload;
  const rest = Object.fromEntries(
    Object.entries(withFloorRemoved).filter(([key]) => !KEYS_REJECTED_WITH_CUSTOM_TIERS.includes(key)),
  ) as typeof withFloorRemoved;
  return { ...rest, ...customTierSetWireFields(customTierSet, classifierLlmConfig, planModeMinTierId) };
};
