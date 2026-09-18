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
  const startsLlmRubric =
    !value.classifier_llm_config ||
    (isForecastClassifier(value.classifier_type) && !isForecastClassifier(classifierType));
  const judgeConfig = value.classifier_llm_config ?? { model: "", timeout_ms: DEFAULT_CLASSIFIER_TIMEOUT_MS };
  const nextValue: ComplexityRouterConfigValue = {
    ...value,
    jev_classifier_config:
      classifierType === "jev" ? value.jev_classifier_config ?? defaultJevClassifierConfig() : undefined,
    classification_prompt: classifierType === "jev" ? undefined : value.classification_prompt,
    classification_examples: classifierType === "jev" ? undefined : value.classification_examples,
    classifier_llm_config: usesLlmClassifier(classifierType)
      ? {
          ...judgeConfig,
          ...(startsLlmRubric && { classification_rubric: NEW_CLASSIFIER_CLASSIFICATION_RUBRIC }),
        }
      : undefined,
    classifier_context_window_size: usesClassifierContext(classifierType)
      ? value.classifier_context_window_size ?? DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE
      : undefined,
    classifier_context_budget_chars: usesClassifierContext(classifierType)
      ? value.classifier_context_budget_chars ?? DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS
      : undefined,
    classifier_context_per_turn_chars: usesClassifierContext(classifierType)
      ? value.classifier_context_per_turn_chars
      : undefined,
    classifier_context_include_assistant_turns: usesClassifierContext(classifierType)
      ? value.classifier_context_include_assistant_turns
      : undefined,
    classifier_fallback: usesClassifierContext(classifierType) ? value.classifier_fallback : undefined,
    heuristic_first_max_tier:
      classifierType === "heuristic_first"
        ? value.heuristic_first_max_tier ?? DEFAULT_HEURISTIC_FIRST_MAX_TIER
        : undefined,
    hybrid_boundary_margin:
      classifierType === "hybrid" ? value.hybrid_boundary_margin ?? DEFAULT_HYBRID_BOUNDARY_MARGIN : undefined,
    ...nonReasoningTierFields(classifierType, value),
  };
  return prepareForecastClassifier(nextValue, classifierType);
};
