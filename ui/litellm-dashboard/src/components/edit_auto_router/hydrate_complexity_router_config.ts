import { defaultJevClassifierConfig, jevClassifierConfigSchema } from "../add_model/jev_classifier_config";
import { capabilitySettingsSchema, fuseSettingsSchema } from "../add_model/forecast_classifier_config";
import type { StoredComplexityRouterConfig } from "../add_model/build_complexity_router_config";
import {
  hydrateBuiltInTiers,
  hydrateCustomTierSet,
  hydratePlanModeMinTier,
  hydrateTierLabels,
} from "../add_model/build_complexity_router_config";
import { hydrateTierModelParams } from "../add_model/complexity_router_tiers";
import { hydrateCustomDimensions } from "../add_model/custom_dimensions";
import {
  hydrateDimensionWeights,
  hydrateReasoningOverrideMinScore,
  hydrateTierBoundaries,
  hydrateTokenThresholds,
} from "../add_model/heuristic_scoring_knobs";
import type { ComplexityRouterConfigValue } from "../add_model/ComplexityRouterConfig";
import { DEFAULT_DEPLOYMENT_AFFINITY, DEFAULT_SESSION_AFFINITY } from "../add_model/ComplexityRouterConfig";
import {
  type ActiveTierSet,
  activeTierRows,
  tierParamsByRowId,
  resolveComplexityDefaultModel,
} from "../add_model/tier_rows";

const isReminderMarkerPair = (input: unknown): input is { open: string; close: string } => {
  if (typeof input !== "object" || input === null) {
    return false;
  }
  if (!("open" in input) || !("close" in input)) {
    return false;
  }
  return typeof input.open === "string" && typeof input.close === "string";
};

const stringList = (input: unknown): string[] | undefined =>
  Array.isArray(input) ? input.filter((item): item is string => typeof item === "string") : undefined;

export const hydratePinnedDefaultModel = (
  storedConfigDefaultModel: unknown,
  litellmParamsDefaultModel: string | null | undefined,
  activeTiers: ActiveTierSet,
): string | undefined => {
  if (typeof storedConfigDefaultModel === "string" && storedConfigDefaultModel.trim()) {
    return storedConfigDefaultModel;
  }
  const tierDerived = resolveComplexityDefaultModel(activeTiers);
  const externalOverride = litellmParamsDefaultModel?.trim();
  return externalOverride && externalOverride !== tierDerived ? externalOverride : undefined;
};

export const hydrateComplexityRouterConfig = (
  parsedConfig: StoredComplexityRouterConfig,
  complexityRouterDefaultModel: string | null | undefined,
): ComplexityRouterConfigValue => {
  const builtIn = hydrateBuiltInTiers(parsedConfig.tiers, parsedConfig.enable_non_reasoning_tier);
  const { tiers: hydratedTiers, enable_non_reasoning_tier } = builtIn;
  const custom_tier_set = hydrateCustomTierSet(parsedConfig);
  const activeTiers = { ...builtIn, custom_tier_set };

  return {
    tiers: hydratedTiers,
    enable_non_reasoning_tier,
    custom_tier_set,
    tier_model_params: tierParamsByRowId(
      hydrateTierModelParams(parsedConfig.tiers, parsedConfig.tier_model_configs),
      activeTierRows(activeTiers),
    ),
    default_model: hydratePinnedDefaultModel(parsedConfig.default_model, complexityRouterDefaultModel, activeTiers),
    plan_mode_min_tier: hydratePlanModeMinTier(parsedConfig.plan_mode_min_tier, custom_tier_set),
    tier_labels: hydrateTierLabels(parsedConfig.tier_labels),
    classifier_type: parsedConfig.classifier_type || "heuristic",
    heuristic_v2_success_threshold:
      typeof parsedConfig.heuristic_v2_success_threshold === "number"
        ? parsedConfig.heuristic_v2_success_threshold
        : undefined,
    capability_classifier_config: capabilitySettingsSchema.safeParse(parsedConfig.capability_classifier_config).data,
    llm_v2_config: fuseSettingsSchema.safeParse(parsedConfig.llm_v2_config).data,
    classifier_llm_config: parsedConfig.classifier_type === "jev" ? undefined : parsedConfig.classifier_llm_config,
    jev_classifier_config:
      parsedConfig.classifier_type === "jev"
        ? jevClassifierConfigSchema.safeParse(parsedConfig.jev_classifier_config ?? {}).data ??
          defaultJevClassifierConfig()
        : undefined,
    classifier_context_window_size:
      typeof parsedConfig.classifier_context_window_size === "number"
        ? parsedConfig.classifier_context_window_size
        : undefined,
    classifier_context_budget_chars:
      typeof parsedConfig.classifier_context_budget_chars === "number"
        ? parsedConfig.classifier_context_budget_chars
        : undefined,
    classifier_context_per_turn_chars:
      typeof parsedConfig.classifier_context_per_turn_chars === "number"
        ? parsedConfig.classifier_context_per_turn_chars
        : undefined,
    classifier_context_include_assistant_turns:
      typeof parsedConfig.classifier_context_include_assistant_turns === "boolean"
        ? parsedConfig.classifier_context_include_assistant_turns
        : undefined,
    classifier_fallback:
      parsedConfig.classifier_fallback === "default_model" || parsedConfig.classifier_fallback === "heuristic"
        ? parsedConfig.classifier_fallback
        : undefined,
    classification_prompt:
      typeof parsedConfig.classification_prompt === "string" && parsedConfig.classification_prompt.trim() !== ""
        ? parsedConfig.classification_prompt
        : undefined,
    classification_examples:
      typeof parsedConfig.classification_examples === "string" && parsedConfig.classification_examples.trim() !== ""
        ? parsedConfig.classification_examples
        : undefined,
    heuristic_first_max_tier:
      typeof parsedConfig.heuristic_first_max_tier === "string" && parsedConfig.heuristic_first_max_tier.trim() !== ""
        ? parsedConfig.heuristic_first_max_tier
        : undefined,
    hybrid_boundary_margin:
      typeof parsedConfig.hybrid_boundary_margin === "number" ? parsedConfig.hybrid_boundary_margin : undefined,
    classification_mode:
      parsedConfig.classification_mode === "user_turn" || parsedConfig.classification_mode === "every_request"
        ? parsedConfig.classification_mode
        : undefined,
    tier_boundaries: hydrateTierBoundaries(parsedConfig.tier_boundaries),
    token_thresholds: hydrateTokenThresholds(parsedConfig.token_thresholds),
    dimension_weights: hydrateDimensionWeights(parsedConfig.dimension_weights),
    custom_dimensions: hydrateCustomDimensions(parsedConfig.custom_dimensions),
    reasoning_override_min_score: hydrateReasoningOverrideMinScore(parsedConfig.reasoning_override_min_score),
    session_affinity:
      typeof parsedConfig.session_affinity === "boolean" ? parsedConfig.session_affinity : DEFAULT_SESSION_AFFINITY,
    session_affinity_ttl_seconds:
      typeof parsedConfig.session_affinity_ttl_seconds === "number" &&
      Number.isFinite(parsedConfig.session_affinity_ttl_seconds)
        ? parsedConfig.session_affinity_ttl_seconds
        : undefined,
    modality_routing: typeof parsedConfig.modality_routing === "boolean" ? parsedConfig.modality_routing : false,
    modality_pin_override:
      typeof parsedConfig.modality_pin_override === "boolean" ? parsedConfig.modality_pin_override : false,
    deployment_affinity:
      typeof parsedConfig.deployment_affinity === "boolean"
        ? parsedConfig.deployment_affinity
        : DEFAULT_DEPLOYMENT_AFFINITY,
    adaptive: parsedConfig.adaptive || false,
    adaptive_weights: parsedConfig.adaptive_weights,
    tier_distance_penalty: parsedConfig.tier_distance_penalty,
    adaptive_eligible: parsedConfig.adaptive_eligible || "all",
    return_raw_model_name: parsedConfig.return_raw_model_name || false,
    enable_context_window_escalation:
      typeof parsedConfig.enable_context_window_escalation === "boolean"
        ? parsedConfig.enable_context_window_escalation
        : undefined,
    context_window_escalation_buffer:
      typeof parsedConfig.context_window_escalation_buffer === "number"
        ? parsedConfig.context_window_escalation_buffer
        : undefined,
    stall_escalation_enabled: parsedConfig.stall_escalation_enabled === true || undefined,
    stall_escalation_window:
      typeof parsedConfig.stall_escalation_window === "number" ? parsedConfig.stall_escalation_window : undefined,
    stall_escalation_repeat_threshold:
      typeof parsedConfig.stall_escalation_repeat_threshold === "number"
        ? parsedConfig.stall_escalation_repeat_threshold
        : undefined,
    code_keywords: stringList(parsedConfig.code_keywords),
    reasoning_keywords: stringList(parsedConfig.reasoning_keywords),
    technical_keywords: stringList(parsedConfig.technical_keywords),
    simple_keywords: stringList(parsedConfig.simple_keywords),
    plan_mode_patterns: stringList(parsedConfig.plan_mode_patterns),
    route_housekeeping_to_cheapest_tier:
      typeof parsedConfig.route_housekeeping_to_cheapest_tier === "boolean"
        ? parsedConfig.route_housekeeping_to_cheapest_tier
        : undefined,
    housekeeping_patterns: stringList(parsedConfig.housekeeping_patterns),
    reminder_markers: Array.isArray(parsedConfig.reminder_markers)
      ? parsedConfig.reminder_markers.filter(isReminderMarkerPair)
      : undefined,
    max_tokens_from_tier_model:
      typeof parsedConfig.max_tokens_from_tier_model === "boolean"
        ? parsedConfig.max_tokens_from_tier_model
        : undefined,
    classifier_plugin_timeout_ms:
      typeof parsedConfig.classifier_plugin_timeout_ms === "number" &&
      Number.isFinite(parsedConfig.classifier_plugin_timeout_ms)
        ? parsedConfig.classifier_plugin_timeout_ms
        : undefined,
  };
};
