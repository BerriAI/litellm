import type { TFunction } from "i18next";
import { asText, readRecord, type GuardrailCriterion, type GuardrailFormValues } from "./GuardrailFormField";
import {
  choiceToLoggingOnlyScope,
  choiceToSkipSystemForCreate,
  choiceToSkipToolForCreate,
  effectiveLoggingOnlyContinue,
  guardrail_provider_map,
  modeIncludesLoggingOnly,
  shouldRenderContentFilterConfigSettings,
  streamScopePayload,
  toModeArray,
  type GuardrailStreamScope,
  type LoggingOnlyScope,
  type LoggingOnlyScopeChoice,
} from "./guardrail_info_helpers";
import type { CompetitorIntentConfig } from "./content_filter/CompetitorIntentConfiguration";
import type { ToolPermissionConfig } from "./tool_permission/ToolPermissionRulesEditor";
import {
  buildDecisionModelParams,
  decisionChecksProblem,
  type DecisionModelCheckDraft,
} from "./decision_model/buildDecisionModelParams";
import type { GuardrailProviderParamsMap } from "./guardrailFormValidation";

interface ContentFilterPatternInput {
  id: string;
  type: "prebuilt" | "custom";
  name: string;
  display_name?: string;
  pattern?: string;
  action: "BLOCK" | "MASK";
}

interface ContentFilterBlockedWordInput {
  id: string;
  keyword: string;
  action: "BLOCK" | "MASK";
  description?: string;
}

interface SelectedContentCategoryInput {
  id: string;
  category: string;
  display_name: string;
  action: "BLOCK" | "MASK";
  severity_threshold: "high" | "medium" | "low";
}

type SkipMessageChoice = "inherit" | "yes" | "no";

const asSkipChoice = (value: unknown): SkipMessageChoice | undefined =>
  value === "inherit" || value === "yes" || value === "no" ? value : undefined;

interface AddGuardrailPayload {
  guardrail_name: string;
  litellm_params: {
    guardrail: string;
    logging_only_scope?: LoggingOnlyScope | null;
    [key: string]: unknown;
  };
  guardrail_info: Record<string, unknown>;
}

type BuildAddGuardrailResult = { ok: true; payload: AddGuardrailPayload } | { ok: false; errorMessage: string };

interface BuildAddGuardrailParams {
  values: GuardrailFormValues;
  selectedProvider: string | null;
  providerParams: GuardrailProviderParamsMap | null;
  selectedEntities: string[];
  selectedActions: Record<string, string>;
  selectedPatterns: ContentFilterPatternInput[];
  blockedWords: ContentFilterBlockedWordInput[];
  selectedContentCategories: SelectedContentCategoryInput[];
  competitorIntentEnabled: boolean;
  competitorIntentConfig: CompetitorIntentConfig | null | undefined;
  decisionChecks: DecisionModelCheckDraft[];
  toolPermissionConfig: ToolPermissionConfig;
  selectedEndpointType: string;
  endSessionAfterNFails: number | undefined;
  onViolation: "warn" | "end_session";
  realtimeViolationMessage: string;
  t: TFunction;
}

export const buildAddGuardrailPayload = ({
  values,
  selectedProvider,
  providerParams,
  selectedEntities,
  selectedActions,
  selectedPatterns,
  blockedWords,
  selectedContentCategories,
  competitorIntentEnabled,
  competitorIntentConfig,
  decisionChecks,
  toolPermissionConfig,
  selectedEndpointType,
  endSessionAfterNFails,
  onViolation,
  realtimeViolationMessage,
  t,
}: BuildAddGuardrailParams): BuildAddGuardrailResult => {
  const providerKey = asText(values.provider);
  const guardrailProvider = guardrail_provider_map[providerKey];

  const guardrailData: AddGuardrailPayload = {
    guardrail_name: asText(values.guardrail_name),
    litellm_params: {
      guardrail: guardrailProvider,
      mode: values.mode,
      default_on: values.default_on,
    },
    guardrail_info: {},
  };

  const streamScope = streamScopePayload(
    toModeArray(values.mode),
    (values.stream_scope_by_mode as Record<string, GuardrailStreamScope> | undefined) ?? {},
  );
  if (streamScope !== undefined) {
    guardrailData.litellm_params.stream_scope = streamScope;
  }

  const skipForCreate = choiceToSkipSystemForCreate(asSkipChoice(values.skip_system_message_choice));
  if (skipForCreate !== undefined) {
    guardrailData.litellm_params.skip_system_message_in_guardrail = skipForCreate;
  }

  const skipToolForCreate = choiceToSkipToolForCreate(asSkipChoice(values.skip_tool_message_choice));
  if (skipToolForCreate !== undefined) {
    guardrailData.litellm_params.skip_tool_message_in_guardrail = skipToolForCreate;
  }

  const loggingOnlyScope = choiceToLoggingOnlyScope(
    values.logging_only_scope_choice as LoggingOnlyScopeChoice | undefined,
  );
  if (modeIncludesLoggingOnly(values.mode) && loggingOnlyScope !== null) {
    guardrailData.litellm_params.logging_only_scope = loggingOnlyScope;
  }
  if (
    modeIncludesLoggingOnly(values.mode) &&
    effectiveLoggingOnlyContinue(
      values.logging_only_scope_choice as LoggingOnlyScopeChoice | undefined,
      values.logging_only_continue_on_input_failure,
    )
  ) {
    guardrailData.litellm_params.logging_only_continue_on_input_failure = true;
  }

  if (providerKey === "PresidioPII" && selectedEntities.length > 0) {
    const piiEntitiesConfig: { [key: string]: string } = {};
    selectedEntities.forEach((entity) => {
      piiEntitiesConfig[entity] = selectedActions[entity] || "MASK";
    });

    guardrailData.litellm_params.pii_entities_config = piiEntitiesConfig;

    if (values.presidio_analyzer_api_base) {
      guardrailData.litellm_params.presidio_analyzer_api_base = values.presidio_analyzer_api_base;
    }
    if (values.presidio_anonymizer_api_base) {
      guardrailData.litellm_params.presidio_anonymizer_api_base = values.presidio_anonymizer_api_base;
    }
  }

  if (shouldRenderContentFilterConfigSettings(providerKey)) {
    const hasCompetitorIntent = competitorIntentEnabled && (competitorIntentConfig?.brand_self?.length ?? 0) > 0;
    const hasContentFilterSelections =
      selectedPatterns.length > 0 || blockedWords.length > 0 || selectedContentCategories.length > 0;
    if (!hasContentFilterSelections && !hasCompetitorIntent) {
      return { ok: false, errorMessage: t("guardrails.addForm.toasts.contentFilterRequired") };
    }

    if (selectedPatterns.length > 0) {
      guardrailData.litellm_params.patterns = selectedPatterns.map((p) => ({
        pattern_type: p.type === "prebuilt" ? "prebuilt" : "regex",
        pattern_name: p.type === "prebuilt" ? p.name : undefined,
        pattern: p.type === "custom" ? p.pattern : undefined,
        name: p.name,
        action: p.action,
      }));
    }
    if (blockedWords.length > 0) {
      guardrailData.litellm_params.blocked_words = blockedWords.map((w) => ({
        keyword: w.keyword,
        action: w.action,
        description: w.description,
      }));
    }
    if (selectedContentCategories.length > 0) {
      guardrailData.litellm_params.categories = selectedContentCategories.map((c) => ({
        category: c.category,
        enabled: true,
        action: c.action,
        severity_threshold: c.severity_threshold || "medium",
      }));
    }
    if (hasCompetitorIntent && competitorIntentConfig) {
      guardrailData.litellm_params.competitor_intent_config = {
        competitor_intent_type: competitorIntentConfig.competitor_intent_type ?? "airline",
        brand_self: competitorIntentConfig.brand_self,
        locations: (competitorIntentConfig.locations?.length ?? 0) > 0 ? competitorIntentConfig.locations : undefined,
        competitors:
          competitorIntentConfig.competitor_intent_type === "generic" &&
          (competitorIntentConfig.competitors?.length ?? 0) > 0
            ? competitorIntentConfig.competitors
            : undefined,
        policy: competitorIntentConfig.policy,
        threshold_high: competitorIntentConfig.threshold_high,
        threshold_medium: competitorIntentConfig.threshold_medium,
        threshold_low: competitorIntentConfig.threshold_low,
      };
    }
  } else if (values.config) {
    try {
      const configObj = JSON.parse(asText(values.config));
      guardrailData.guardrail_info = configObj;
    } catch {
      return { ok: false, errorMessage: t("guardrails.addForm.toasts.invalidJson") };
    }
  }

  if (guardrailProvider === "llm_as_a_judge") {
    const criteria: GuardrailCriterion[] = (values.criteria as GuardrailCriterion[] | undefined) ?? [];
    if (criteria.length === 0) {
      return { ok: false, errorMessage: t("guardrails.addForm.toasts.criterionRequired") };
    }
    const weightTotal = criteria.reduce((sum, c) => sum + (Number(c?.weight) || 0), 0);
    if (weightTotal !== 100) {
      return {
        ok: false,
        errorMessage: t("guardrails.addForm.toasts.criterionWeights", { total: weightTotal }),
      };
    }
    guardrailData.litellm_params.judge_model = values.judge_model;
    guardrailData.litellm_params.overall_threshold = values.overall_threshold ?? 80;
    guardrailData.litellm_params.on_failure = values.on_failure ?? "block";
    guardrailData.litellm_params.criteria = criteria.map((c) => ({
      name: c.name,
      weight: Number(c.weight),
      description: c.description || "",
    }));
  }

  if (guardrailProvider === "decision_model") {
    const questionsProblem = decisionChecksProblem(decisionChecks, t);
    if (questionsProblem) {
      return { ok: false, errorMessage: questionsProblem };
    }
    const decisionParams = buildDecisionModelParams(asText(values.decision_model), decisionChecks);
    guardrailData.litellm_params.decision_model = decisionParams.decision_model;
    guardrailData.litellm_params.checks = decisionParams.checks;
  }

  if (guardrailProvider === "tool_permission") {
    if (toolPermissionConfig.rules.length === 0) {
      return { ok: false, errorMessage: t("guardrails.addForm.toasts.toolRuleRequired") };
    }
    guardrailData.litellm_params.rules = toolPermissionConfig.rules;
    guardrailData.litellm_params.default_action = toolPermissionConfig.default_action;
    guardrailData.litellm_params.on_disallowed_action = toolPermissionConfig.on_disallowed_action;
    if (toolPermissionConfig.violation_message_template) {
      guardrailData.litellm_params.violation_message_template = toolPermissionConfig.violation_message_template;
    }
  }

  if (shouldRenderContentFilterConfigSettings(providerKey)) {
    if (endSessionAfterNFails !== undefined && endSessionAfterNFails > 0) {
      guardrailData.litellm_params.end_session_after_n_fails = endSessionAfterNFails;
    }
    if (onViolation && selectedEndpointType === "realtime") {
      guardrailData.litellm_params.on_violation = onViolation;
    }
    if (realtimeViolationMessage.trim()) {
      guardrailData.litellm_params.realtime_violation_message = realtimeViolationMessage.trim();
    }
  }

  const providerOwnsLitellmParams = ["llm_as_a_judge", "decision_model"].includes(guardrailProvider);
  if (providerParams && selectedProvider && !providerOwnsLitellmParams) {
    const normalizedProviderKey = guardrail_provider_map[selectedProvider]?.toLowerCase();
    const providerSpecificParams = normalizedProviderKey ? providerParams[normalizedProviderKey] || {} : {};

    const allowedParams = new Set<string>();

    Object.keys(providerSpecificParams).forEach((paramName) => {
      if (paramName !== "optional_params") {
        allowedParams.add(paramName);
      }
    });

    if (providerSpecificParams.optional_params && providerSpecificParams.optional_params.fields) {
      Object.keys(providerSpecificParams.optional_params.fields).forEach((paramName) => {
        allowedParams.add(paramName);
      });
    }

    allowedParams.forEach((paramName) => {
      const directValue = values[paramName];
      const paramValue =
        directValue === undefined || directValue === null || directValue === ""
          ? readRecord(values.optional_params, paramName)
          : directValue;

      if (paramValue !== undefined && paramValue !== null && paramValue !== "") {
        guardrailData.litellm_params[paramName] = paramValue;
      }
    });
  }

  return { ok: true, payload: guardrailData };
};
