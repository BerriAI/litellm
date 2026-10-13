import {
  type ClassifierType,
  type ComplexityRouterConfigValue,
  DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS,
  DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
  DEFAULT_CLASSIFIER_TIMEOUT_MS,
  DEFAULT_HEURISTIC_FIRST_MAX_TIER,
  DEFAULT_HYBRID_BOUNDARY_MARGIN,
  NEW_CLASSIFIER_CLASSIFICATION_RUBRIC,
  usesLlmClassifier,
  usesClassifierContext,
} from "./ComplexityRouterConfig";
import { defaultJevClassifierConfig } from "./jev_classifier_config";
import { isForecastClassifier, prepareForecastClassifier } from "./forecast_classifier_config";
import { nonReasoningTierFields } from "./nonReasoningTierFields";

export const transitionClassifierType = (
  value: ComplexityRouterConfigValue,
  classifierType: ClassifierType,
): ComplexityRouterConfigValue => {
  const hasComplexityDraft =
    value.classifier_llm_config?.classification_rubric !== undefined ||
    [value.classifier_type, ...(value.previous_classifier_types ?? [])].some(
      (type) => usesLlmClassifier(type) && !isForecastClassifier(type),
    );
  const startsLlmRubric = !value.classifier_llm_config || !hasComplexityDraft;
  const judgeConfig = value.classifier_llm_config ?? { model: "", timeout_ms: DEFAULT_CLASSIFIER_TIMEOUT_MS };
  const nextValue: ComplexityRouterConfigValue = {
    ...value,
    previous_classifier_types: [
      value.classifier_type,
      ...(value.previous_classifier_types ?? []).filter((type) => type !== value.classifier_type),
    ],
    jev_classifier_config:
      classifierType === "jev"
        ? value.jev_classifier_config ?? defaultJevClassifierConfig()
        : value.jev_classifier_config,
    classifier_llm_config: usesLlmClassifier(classifierType)
      ? {
          ...judgeConfig,
          ...(startsLlmRubric && { classification_rubric: NEW_CLASSIFIER_CLASSIFICATION_RUBRIC }),
        }
      : value.classifier_llm_config,
    classifier_context_window_size: usesClassifierContext(classifierType)
      ? value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE
      : value.classifier_context_window_size,
    classifier_context_budget_chars: usesClassifierContext(classifierType)
      ? value.classifier_context_budget_chars ?? DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS
      : value.classifier_context_budget_chars,
    heuristic_first_max_tier:
      classifierType === "heuristic_first"
        ? value.heuristic_first_max_tier ?? DEFAULT_HEURISTIC_FIRST_MAX_TIER
        : value.heuristic_first_max_tier,
    hybrid_boundary_margin:
      classifierType === "hybrid"
        ? value.hybrid_boundary_margin ?? DEFAULT_HYBRID_BOUNDARY_MARGIN
        : value.hybrid_boundary_margin,
    ...nonReasoningTierFields(classifierType, value),
  };
  return prepareForecastClassifier(nextValue, classifierType);
};
