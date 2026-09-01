import { KeywordTierRule } from "./KeywordTierRules";
import {
  type CustomTierSet,
  type TierRow,
  CUSTOM_TIER_OMITTED_KEYS,
  activeTierName,
  sameTierIdentity,
  tierDefinitionsFromRows,
  tierRowById,
  tierRowByName,
} from "./tier_rows";
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
  ClassificationMode,
  ClassifierFallback,
  ClassifierLLMConfig,
  ClassifierType,
  ComplexityTierLabels,
  DEFAULT_CLASSIFICATION_MODE,
  ComplexityRouterConfigValue,
  ComplexityTiers,
  DimensionWeights,
  TIER_KEYS,
  effectiveClassifierType,
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
  customTierSet?: CustomTierSet;
  defaultModel: string | undefined;
  planModeMinTier: string | undefined;
  tierLabels: ComplexityTierLabels | undefined;
  classifierType: ClassifierType;
  classifierLlmConfig: ClassifierLLMConfig | undefined;
  classifierContextWindowSize: number | undefined;
  classifierContextBudgetChars: number | undefined;
  classifierContextIncludeAssistantTurns: boolean | undefined;
  classifierFallback: ClassifierFallback | undefined;
  classificationPrompt: string | undefined;
  heuristicFirstMaxTier: string | undefined;
  classificationMode: ClassificationMode | undefined;
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
  enableContextWindowEscalation?: boolean;
  contextWindowEscalationBuffer?: number;
}

/**
 * The message to surface when the dry-run rejects a save, or null to let it through.
 *
 * Gated on `valid` alone. The verdict's `valid` is derived from `error` server side today, but the
 * two arrive as independent fields, so reading `error` as the gate would let a rejection whose
 * message is missing or blank through to the write and back as a raw 400. A transport failure fails
 * open as `{valid: true}`, which this passes, leaving the write gate authoritative.
 */
export const dryRunRejection = (verdict: { valid: boolean; error?: string | null }): string | null =>
  verdict.valid ? null : verdict.error?.trim() || "The proxy rejected this auto-router configuration";

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
  classifier_context_budget_chars?: number;
  classifier_context_per_turn_chars?: number;
  classifier_context_include_assistant_turns?: boolean;
  classifier_fallback?: ClassifierFallback;
  classification_prompt?: string;
  heuristic_first_max_tier?: string;
  classification_mode: ClassificationMode;
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
  enable_context_window_escalation?: boolean;
  context_window_escalation_buffer?: number;
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
// built-in set, which is why it keeps its own !defaultModel guard after deriving.
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

// The orphan check compares exactly, not casefold: _validate_keyword_rule_tiers is exact
// membership, so a rule left pointing at a differently cased name would clear a gate the
// backend then rejects.
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

// An edited tier set forces the LLM classifier, so the model requirement follows the EFFECTIVE type.
// Both forms' submit gates and their submit handlers read this one answer so they cannot drift.
export const getClassifierModelError = (
  config: Pick<ComplexityRouterConfigValue, "custom_tier_set" | "classifier_type" | "classifier_llm_config">,
): string | null => {
  if (!usesLlmClassifier(effectiveClassifierType(config)) || config.classifier_llm_config?.model) return null;
  return config.custom_tier_set
    ? "Please select a classifier model: an edited tier set routes with the LLM classifier"
    : "Please select a classifier model, or switch back to Heuristic";
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

export const customTierWireFields = (
  customTierSet: CustomTierSet,
  classifierLlmConfig: ClassifierLLMConfig | undefined,
  planModeMinTierId: string | undefined,
  classificationPrompt: string | undefined,
): Partial<ComplexityRouterConfigPayload> => {
  const rows = customTierSet.tiers;
  const fallback = tierRowById(rows, customTierSet.fallback_tier_id);
  const floor = tierRowById(rows, planModeMinTierId);
  return {
    tiers: Object.fromEntries(rows.map((row) => [activeTierName(row), row.models])),
    tier_definitions: tierDefinitionsFromRows(rows),
    ...(fallback && { fallback_tier: activeTierName(fallback) }),
    classifier_type: "llm",
    // Rebuilt from the two fields an edited tier set allows. The backend rejects system_prompt and
    // classification_rubric beside tier_definitions, and both live inside this object rather than at
    // the top level the omit list covers. The opening instructions ride classification_prompt below.
    ...(classifierLlmConfig && {
      classifier_llm_config: { model: classifierLlmConfig.model, timeout_ms: classifierLlmConfig.timeout_ms },
    }),
    session_affinity: false,
    ...(classificationPrompt?.trim() && { classification_prompt: classificationPrompt.trim() }),
    ...(floor && { plan_mode_min_tier: activeTierName(floor) }),
  };
};

// plan_mode_min_tier rides the strip list because the base payload carries it as a row id;
// customTierWireFields re-emits it as the row's name, and an unresolvable floor stays off.
const CUSTOM_TIER_STRIPPED_KEYS: readonly string[] = [...CUSTOM_TIER_OMITTED_KEYS, "plan_mode_min_tier"];

export const hydrateCustomTierSet = (parsedConfig: {
  tier_definitions?: unknown;
  fallback_tier?: unknown;
  tiers?: unknown;
}): CustomTierSet | undefined => {
  if (!Array.isArray(parsedConfig.tier_definitions) || parsedConfig.tier_definitions.length === 0) return undefined;
  const storedTiers =
    typeof parsedConfig.tiers === "object" && parsedConfig.tiers !== null && !Array.isArray(parsedConfig.tiers)
      ? Object.entries(parsedConfig.tiers as Record<string, unknown>)
      : [];
  const rows = parsedConfig.tier_definitions.flatMap((entry, index): TierRow[] => {
    if (typeof entry !== "object" || entry === null) return [];
    const { name, description } = entry as { name?: unknown; description?: unknown };
    if (typeof name !== "string" || !name.trim()) return [];
    return [
      {
        id: TIER_KEYS.find((tier) => sameTierIdentity(tier, name)) ?? `stored-${index}`,
        name: name.trim(),
        definition: typeof description === "string" ? description.trim() : "",
        models: normalizeTierModels(storedTiers.find(([tier]) => sameTierIdentity(tier, name))?.[1]),
      },
    ];
  });
  if (rows.length === 0) return undefined;
  const storedFallback = typeof parsedConfig.fallback_tier === "string" ? parsedConfig.fallback_tier : "";
  return { tiers: rows, fallback_tier_id: tierRowByName(rows, storedFallback)?.id ?? "" };
};

// Ids are session-ephemeral, so a stored floor hydrates by name; unresolvable means off, the same
// rule the editor and the wire apply.
export const hydratePlanModeMinTier = (
  stored: unknown,
  customTierSet: CustomTierSet | undefined,
): string | undefined => {
  if (typeof stored !== "string" || !stored.trim()) return undefined;
  if (!customTierSet) return stored;
  return tierRowByName(customTierSet.tiers, stored)?.id;
};

const classifierWireFields = (
  effectiveType: ClassifierType,
  {
    classifierLlmConfig,
    classifierFallback,
    heuristicFirstMaxTier,
    classifierContextWindowSize,
    classifierContextBudgetChars,
    classifierContextIncludeAssistantTurns,
  }: Pick<
    BuildComplexityRouterConfigParams,
    | "classifierLlmConfig"
    | "classifierFallback"
    | "heuristicFirstMaxTier"
    | "classifierContextWindowSize"
    | "classifierContextBudgetChars"
    | "classifierContextIncludeAssistantTurns"
  >,
): Partial<ComplexityRouterConfigPayload> => ({
  ...(usesLlmClassifier(effectiveType) &&
    classifierLlmConfig && { classifier_llm_config: normalizeClassifierLlmConfig(classifierLlmConfig) }),
  ...(usesLlmClassifier(effectiveType) &&
    classifierFallback !== undefined && { classifier_fallback: classifierFallback }),
  ...(effectiveType === "heuristic_first" &&
    heuristicFirstMaxTier?.trim() && { heuristic_first_max_tier: heuristicFirstMaxTier }),
  ...(usesLlmClassifier(effectiveType) &&
    classifierContextWindowSize !== undefined && {
      classifier_context_window_size: classifierContextWindowSize,
    }),
  ...(usesLlmClassifier(effectiveType) &&
    classifierContextBudgetChars !== undefined && {
      classifier_context_budget_chars: classifierContextBudgetChars,
    }),
  ...(usesLlmClassifier(effectiveType) &&
    classifierContextIncludeAssistantTurns !== undefined && {
      classifier_context_include_assistant_turns: classifierContextIncludeAssistantTurns,
    }),
});

export const buildComplexityRouterConfig = ({
  tiers,
  customTierSet,
  defaultModel,
  planModeMinTier,
  tierLabels,
  classifierType,
  classifierLlmConfig,
  classifierContextWindowSize,
  classifierContextBudgetChars,
  classifierContextIncludeAssistantTurns,
  classifierFallback,
  classificationPrompt,
  heuristicFirstMaxTier,
  classificationMode,
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
  enableContextWindowEscalation,
  contextWindowEscalationBuffer,
}: BuildComplexityRouterConfigParams): ComplexityRouterConfigPayload => {
  const serializedTierModelConfigs = customTierSet
    ? serializeTierModelConfigs(
        Object.fromEntries(customTierSet.tiers.map((row) => [activeTierName(row), row.models])),
        Object.fromEntries(customTierSet.tiers.map((row) => [activeTierName(row), tierModelParams?.[row.id] ?? {}])),
      )
    : serializeTierModelConfigs(tiers, tierModelParams);
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
  const classifierInputs = {
    classifierLlmConfig,
    classifierFallback,
    heuristicFirstMaxTier,
    classifierContextWindowSize,
    classifierContextBudgetChars,
    classifierContextIncludeAssistantTurns,
  };
  // An edited tier set forces the LLM classifier, so llm-only inputs must survive a classifier_type
  // the form never rewrote. The UI gates the same controls on this, not on the raw value.
  const effectiveType: ClassifierType = customTierSet ? "llm" : classifierType;

  const payload: ComplexityRouterConfigPayload = {
    tiers,
    ...(serializedTierModelConfigs && { tier_model_configs: serializedTierModelConfigs }),
    ...(defaultModel?.trim() && { default_model: defaultModel }),
    ...(planModeMinTier?.trim() && { plan_mode_min_tier: planModeMinTier }),
    ...(cleanedTierLabels && { tier_labels: cleanedTierLabels }),
    classifier_type: classifierType,
    ...classifierWireFields(effectiveType, classifierInputs),
    classification_mode: classificationMode ?? DEFAULT_CLASSIFICATION_MODE,
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
    ...(enableContextWindowEscalation !== undefined && {
      enable_context_window_escalation: enableContextWindowEscalation,
    }),
    ...(contextWindowEscalationBuffer !== undefined && {
      context_window_escalation_buffer: contextWindowEscalationBuffer,
    }),
    ...scorerKnobs,
  };
  if (!customTierSet) return payload;
  const kept = Object.fromEntries(
    Object.entries(payload).filter(([key]) => !CUSTOM_TIER_STRIPPED_KEYS.includes(key)),
  ) as ComplexityRouterConfigPayload;
  return {
    ...kept,
    ...customTierWireFields(customTierSet, classifierLlmConfig, planModeMinTier, classificationPrompt),
  };
};
