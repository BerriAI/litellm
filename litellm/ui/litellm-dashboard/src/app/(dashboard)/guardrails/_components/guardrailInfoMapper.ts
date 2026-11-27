import { asText, readRecord, type GuardrailFormValues } from "./GuardrailFormField";
import {
  getLoggingOnlyScopeUpdate,
  guardrail_provider_map,
  loggingOnlyContinueFromParams,
  loggingOnlyScopeToChoice,
  skipSystemMessageToChoice,
  skipToolMessageToChoice,
  streamScopeByModeFromConfig,
  streamScopeForUpdate,
  toModeArray,
  type GuardrailStreamScope,
  type SkipSystemMessageChoice,
} from "./guardrail_info_helpers";
import { formatContentFilterDataForAPI } from "./content_filter/ContentFilterManager";
import type { ToolPermissionConfig } from "./tool_permission/ToolPermissionRulesEditor";
import type { GuardrailProviderParamsMap } from "./guardrailFormValidation";

interface ContentFilterSnapshot {
  patterns: any[];
  blockedWords: any[];
  categories: any[];
  competitorIntentEnabled?: boolean;
  competitorIntentConfig?: any;
}

export const guardrailInfoToFormValues = (guardrailData: any): GuardrailFormValues => {
  const storedParams = guardrailData.litellm_params;
  const values: GuardrailFormValues = {
    guardrail_name: guardrailData.guardrail_name,
    default_on: storedParams?.default_on,
    logging_only_scope_choice: loggingOnlyScopeToChoice(storedParams?.logging_only_scope),
    logging_only_continue_on_input_failure: loggingOnlyContinueFromParams(storedParams),
    skip_system_message_choice: skipSystemMessageToChoice(storedParams?.skip_system_message_in_guardrail),
    skip_tool_message_choice: skipToolMessageToChoice(storedParams?.skip_tool_message_in_guardrail),
    stream_scope_by_mode: streamScopeByModeFromConfig(storedParams?.stream_scope, toModeArray(storedParams?.mode)),
    guardrail_info: guardrailData.guardrail_info ? JSON.stringify(guardrailData.guardrail_info, null, 2) : "",
  };
  if (storedParams?.optional_params) {
    values.optional_params = storedParams.optional_params;
  }
  return values;
};

interface BuildGuardrailUpdateParams {
  values: GuardrailFormValues;
  guardrailData: any;
  selectedPiiEntities: string[];
  selectedPiiActions: Record<string, string>;
  hasUnsavedContentFilterChanges: boolean;
  contentFilterData: ContentFilterSnapshot;
  toolPermissionConfig: ToolPermissionConfig;
  toolPermissionDirty: boolean;
  guardrailProviderSpecificParams: GuardrailProviderParamsMap | null;
}

export const buildGuardrailUpdatePayload = ({
  values,
  guardrailData,
  selectedPiiEntities,
  selectedPiiActions,
  hasUnsavedContentFilterChanges,
  contentFilterData,
  toolPermissionConfig,
  toolPermissionDirty,
  guardrailProviderSpecificParams,
}: BuildGuardrailUpdateParams): Record<string, any> | null => {
  const { logging_only_scope_choice: scopeChoice, logging_only_continue_on_input_failure: continueFlag } = values;
  const updateData: any = {
    litellm_params: getLoggingOnlyScopeUpdate(guardrailData.litellm_params, scopeChoice, continueFlag),
  };

  if (values.guardrail_name !== guardrailData.guardrail_name) {
    updateData.guardrail_name = values.guardrail_name;
  }

  if (values.default_on !== guardrailData.litellm_params?.default_on) {
    updateData.litellm_params.default_on = values.default_on;
  }

  const modes = toModeArray(guardrailData.litellm_params?.mode);
  const nextStreamScope = streamScopeForUpdate(
    modes,
    (values.stream_scope_by_mode as Record<string, GuardrailStreamScope> | undefined) ?? {},
    guardrailData.litellm_params?.stream_scope,
  );
  if (nextStreamScope !== undefined) {
    updateData.litellm_params.stream_scope = nextStreamScope;
  }

  const prevSkipChoice = skipSystemMessageToChoice(guardrailData.litellm_params?.skip_system_message_in_guardrail);
  const nextSkipChoice = values.skip_system_message_choice as SkipSystemMessageChoice | undefined;
  if (nextSkipChoice !== undefined && nextSkipChoice !== prevSkipChoice) {
    if (nextSkipChoice === "inherit") {
      updateData.litellm_params.skip_system_message_in_guardrail = null;
    } else if (nextSkipChoice === "yes") {
      updateData.litellm_params.skip_system_message_in_guardrail = true;
    } else {
      updateData.litellm_params.skip_system_message_in_guardrail = false;
    }
  }

  const prevSkipToolChoice = skipToolMessageToChoice(guardrailData.litellm_params?.skip_tool_message_in_guardrail);
  const nextSkipToolChoice = values.skip_tool_message_choice as SkipSystemMessageChoice | undefined;
  if (nextSkipToolChoice !== undefined && nextSkipToolChoice !== prevSkipToolChoice) {
    if (nextSkipToolChoice === "inherit") {
      updateData.litellm_params.skip_tool_message_in_guardrail = null;
    } else if (nextSkipToolChoice === "yes") {
      updateData.litellm_params.skip_tool_message_in_guardrail = true;
    } else {
      updateData.litellm_params.skip_tool_message_in_guardrail = false;
    }
  }

  const originalGuardrailInfo = guardrailData.guardrail_info;
  const newGuardrailInfo = values.guardrail_info ? JSON.parse(asText(values.guardrail_info)) : undefined;
  if (JSON.stringify(originalGuardrailInfo) !== JSON.stringify(newGuardrailInfo)) {
    updateData.guardrail_info = newGuardrailInfo;
  }

  const originalPiiConfig = guardrailData.litellm_params?.pii_entities_config || {};
  const newPiiEntitiesConfig: { [key: string]: string } = {};

  selectedPiiEntities.forEach((entity) => {
    newPiiEntitiesConfig[entity] = selectedPiiActions[entity] || "MASK";
  });

  if (JSON.stringify(originalPiiConfig) !== JSON.stringify(newPiiEntitiesConfig)) {
    updateData.litellm_params.pii_entities_config = newPiiEntitiesConfig;
  }

  if (guardrailData.litellm_params?.guardrail === "litellm_content_filter" && hasUnsavedContentFilterChanges) {
    const formattedData = formatContentFilterDataForAPI(
      contentFilterData.patterns || [],
      contentFilterData.blockedWords || [],
      contentFilterData.categories || [],
      contentFilterData.competitorIntentEnabled,
      contentFilterData.competitorIntentConfig,
    );

    updateData.litellm_params.patterns = formattedData.patterns;
    updateData.litellm_params.blocked_words = formattedData.blocked_words;
    updateData.litellm_params.categories = formattedData.categories;
    updateData.litellm_params.competitor_intent_config = formattedData.competitor_intent_config ?? null;
  }

  if (guardrailData.litellm_params?.guardrail === "tool_permission") {
    const originalRules = guardrailData.litellm_params?.rules || [];
    const currentRules = toolPermissionConfig.rules || [];
    const rulesChanged = JSON.stringify(originalRules) !== JSON.stringify(currentRules);

    const originalDefault = (guardrailData.litellm_params?.default_action || "deny").toLowerCase();
    const currentDefault = (toolPermissionConfig.default_action || "deny").toLowerCase();
    const defaultChanged = originalDefault !== currentDefault;

    const originalOnDisallowed = (guardrailData.litellm_params?.on_disallowed_action || "block").toLowerCase();
    const currentOnDisallowed = (toolPermissionConfig.on_disallowed_action || "block").toLowerCase();
    const onDisallowedChanged = originalOnDisallowed !== currentOnDisallowed;

    const originalMessage = guardrailData.litellm_params?.violation_message_template || "";
    const currentMessage = toolPermissionConfig.violation_message_template || "";
    const messageChanged = originalMessage !== currentMessage;

    if (toolPermissionDirty || rulesChanged || defaultChanged || onDisallowedChanged || messageChanged) {
      updateData.litellm_params.rules = currentRules;
      updateData.litellm_params.default_action = currentDefault;
      updateData.litellm_params.on_disallowed_action = currentOnDisallowed;
      updateData.litellm_params.violation_message_template = currentMessage || null;
    }
  }

  const currentProvider = Object.keys(guardrail_provider_map).find(
    (key) => guardrail_provider_map[key] === guardrailData.litellm_params?.guardrail,
  );

  const isToolPermissionGuardrail = guardrailData.litellm_params?.guardrail === "tool_permission";
  if (guardrailProviderSpecificParams && currentProvider && !isToolPermissionGuardrail) {
    const providerKey = guardrail_provider_map[currentProvider]?.toLowerCase();
    const providerSpecificParams = guardrailProviderSpecificParams[providerKey] || {};

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
      if (paramName === "patterns" || paramName === "blocked_words" || paramName === "categories") {
        return;
      }
      const directValue = values[paramName];
      const paramValue =
        directValue === undefined || directValue === null || directValue === ""
          ? readRecord(values.optional_params, paramName)
          : directValue;

      const originalValue = guardrailData.litellm_params?.[paramName];
      const hasChanged = JSON.stringify(paramValue) !== JSON.stringify(originalValue);

      if (hasChanged) {
        if (paramValue !== undefined && paramValue !== null && paramValue !== "") {
          updateData.litellm_params[paramName] = paramValue;
        } else if (originalValue !== undefined && originalValue !== null && originalValue !== "") {
          updateData.litellm_params[paramName] = null;
        }
      }
    });
  }

  if (Object.keys(updateData.litellm_params).length === 0) {
    delete updateData.litellm_params;
  }

  if (Object.keys(updateData).length === 0) {
    return null;
  }

  return updateData;
};
