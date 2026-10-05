import type { BuildComplexityRouterConfigParams } from "./build_complexity_router_config";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import {
  DEFAULT_ADAPTIVE_WEIGHTS,
  DEFAULT_DEPLOYMENT_AFFINITY,
  DEFAULT_SESSION_AFFINITY,
  DEFAULT_TIER_DISTANCE_PENALTY,
} from "./ComplexityRouterConfig";

export const builderParamsFromValue = (
  value: ComplexityRouterConfigValue,
): Omit<
  BuildComplexityRouterConfigParams,
  | "customTechnicalKeywords"
  | "keywordTierRules"
  | "semanticMatchingEnabled"
  | "embeddingModel"
  | "matchThreshold"
  | "escalationKeywords"
> => ({
  tiers: value.tiers,
  enableNonReasoningTier: value.enable_non_reasoning_tier,
  customTierSet: value.custom_tier_set,
  defaultModel: value.default_model,
  planModeMinTier: value.plan_mode_min_tier,
  classificationPrompt: value.classification_prompt,
  classificationExamples: value.classification_examples,
  heuristicFirstMaxTier: value.heuristic_first_max_tier,
  hybridBoundaryMargin: value.hybrid_boundary_margin,
  classificationMode: value.classification_mode,
  tierLabels: value.tier_labels,
  classifierType: value.classifier_type,
  jevClassifierConfig: value.jev_classifier_config,
  heuristicV2SuccessThreshold: value.heuristic_v2_success_threshold,
  capabilityClassifierConfig: value.capability_classifier_config,
  llmV2Config: value.llm_v2_config,
  classifierLlmConfig: value.classifier_llm_config,
  classifierContextWindowSize: value.classifier_context_window_size,
  classifierContextBudgetChars: value.classifier_context_budget_chars,
  classifierContextPerTurnChars: value.classifier_context_per_turn_chars,
  classifierContextIncludeAssistantTurns: value.classifier_context_include_assistant_turns,
  classifierFallback: value.classifier_fallback,
  sessionAffinity: value.session_affinity ?? DEFAULT_SESSION_AFFINITY,
  sessionAffinityTtlSeconds: value.session_affinity_ttl_seconds,
  modalityRouting: value.modality_routing ?? false,
  modalityPinOverride: value.modality_pin_override ?? false,
  deploymentAffinity: value.deployment_affinity ?? DEFAULT_DEPLOYMENT_AFFINITY,
  adaptive: value.adaptive ?? false,
  adaptiveWeights: value.adaptive_weights ?? DEFAULT_ADAPTIVE_WEIGHTS,
  tierDistancePenalty: value.tier_distance_penalty ?? DEFAULT_TIER_DISTANCE_PENALTY,
  adaptiveEligible: value.adaptive_eligible ?? "all",
  returnRawModelName: value.return_raw_model_name ?? false,
  tierBoundaries: value.tier_boundaries,
  tokenThresholds: value.token_thresholds,
  dimensionWeights: value.dimension_weights,
  customDimensions: value.custom_dimensions,
  reasoningOverrideMinScore: value.reasoning_override_min_score,
  tierModelParams: value.tier_model_params,
  enableContextWindowEscalation: value.enable_context_window_escalation,
  contextWindowEscalationBuffer: value.context_window_escalation_buffer,
  stallEscalationEnabled: value.stall_escalation_enabled,
  stallEscalationWindow: value.stall_escalation_window,
  stallEscalationRepeatThreshold: value.stall_escalation_repeat_threshold,
  codeKeywords: value.code_keywords,
  reasoningKeywords: value.reasoning_keywords,
  technicalKeywords: value.technical_keywords,
  simpleKeywords: value.simple_keywords,
  planModePatterns: value.plan_mode_patterns,
  routeHousekeepingToCheapestTier: value.route_housekeeping_to_cheapest_tier,
  housekeepingPatterns: value.housekeeping_patterns,
  reminderMarkers: value.reminder_markers,
  maxTokensFromTierModel: value.max_tokens_from_tier_model,
  classifierPluginTimeoutMs: value.classifier_plugin_timeout_ms,
});
