import {
  type ClassifierType,
  type ComplexityRouterConfigValue,
  DEFAULT_CLASSIFIER_CONTEXT_BUDGET_CHARS,
  DEFAULT_CLASSIFIER_CONTEXT_WINDOW_SIZE,
  DEFAULT_CLASSIFIER_TIMEOUT_MS,
  NEW_CLASSIFIER_CLASSIFICATION_RUBRIC,
  usesLlmClassifier,
  usesClassifierContext,
} from "./ComplexityRouterConfig";
import { defaultJevClassifierConfig } from "./jev_classifier_config";

export const transitionClassifierType = (
  value: ComplexityRouterConfigValue,
  classifierType: ClassifierType,
): ComplexityRouterConfigValue => {
  const startsLlmRubric = !value.classifier_llm_config;
  const judgeConfig = value.classifier_llm_config ?? { model: "", timeout_ms: DEFAULT_CLASSIFIER_TIMEOUT_MS };
  const nextValue: ComplexityRouterConfigValue = {
    ...value,
    classifier_type: classifierType,
    jev_classifier_config:
      classifierType === "jev" ? value.jev_classifier_config ?? defaultJevClassifierConfig() : undefined,
    classification_prompt: classifierType === "jev" ? undefined : value.classification_prompt,
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
  };
  return nextValue;
};
