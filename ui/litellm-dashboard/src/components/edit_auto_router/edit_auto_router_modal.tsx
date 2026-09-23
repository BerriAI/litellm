import { AutoRouterAvailabilityContext, useAutoRouterAvailability } from "../add_model/AutoRouterAvailability";
import AutoRouterClassifierTabs from "../add_model/AutoRouterClassifierTabs";
import { usesClassifierContext } from "../add_model/classifier_types";
export type { StoredComplexityRouterConfig } from "../add_model/build_complexity_router_config";
import { getForecastConfigError, isForecastClassifier } from "../add_model/forecast_classifier_config";
import React, { useEffect, useMemo, useState } from "react";
import {
  complexityRouterSchema,
  semanticRouterSchema,
  EMPTY_FORM_VALUES,
  type EditAutoRouterFormValues,
} from "./editAutoRouterFormSchema";
import { toast } from "@/lib/toast";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import AccessGroupTagsCombobox from "../add_model/AccessGroupTagsCombobox";
import ModelChoiceCombobox, { type ModelChoice } from "../add_model/ModelChoiceCombobox";
import { modelAvailableCall, modelPatchUpdateCall, validateAutoRouterConfig } from "../networking";
import { fetchAutoRouterModels, fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import RouterConfigBuilder, { type RouterConfig, serializeRouterConfig } from "../add_model/RouterConfigBuilder";
import {
  CUSTOM_TIER_OMITTED_KEYS,
  activeTierRows,
  getCustomTierRowsError,
  resolveComplexityDefaultModel,
} from "../add_model/tier_rows";
import { isComplexityRouter } from "../add_model/auto_router_strategies";
import {
  type BuildComplexityRouterConfigParams,
  type StoredComplexityRouterConfig,
  buildComplexityRouterConfig,
  getClassifierModelError,
  getHeuristicV2SuccessThresholdError,
  getReminderMarkersError,
  getClassifierPluginTimeoutError,
  getClassifierReasoningEffortError,
  getKeywordTierRulesError,
  getMissingTiersError,
  getSemanticConfigError,
  getPlanModeTierError,
  getTierLabelsError,
  dryRunRejection,
} from "../add_model/build_complexity_router_config";
import { KeywordTierRule } from "../add_model/KeywordTierRules";
import { DEFAULT_MATCH_THRESHOLD } from "../add_model/SemanticKeywordMatching";
import {
  type AutoRouterCompressionState,
  buildAutoRouterCompressionPatch,
  DEFAULT_AUTO_ROUTER_COMPRESSION,
  hydrateAutoRouterCompression,
} from "../add_model/buildAutoRouterCompression";
import { hydrateKeywordTierRules } from "../add_model/complexity_router_keywords";
import { customDimensionsError } from "../add_model/custom_dimensions";
import ComplexityRouterConfig, {
  ComplexityRouterConfigValue,
  effectiveClassifierType,
  heuristicScoringRole,
} from "../add_model/ComplexityRouterConfig";
import { builderParamsFromValue } from "../add_model/complexity_router_builder_params";
import { hydrateComplexityRouterConfig, hydratePinnedDefaultModel } from "./hydrate_complexity_router_config";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

interface EditAutoRouterModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: (updatedModel: any) => void;
  modelData: any;
  accessToken: string;
  userRole: string;
  isMemberManaged?: boolean;
}

// Keys this modal rewrites from its own form state on save. Anything absent from this set is
// carried through untouched from the stored config, so a key only belongs here once the modal
// actually renders a control that can set it.
export { hydrateComplexityRouterConfig, hydratePinnedDefaultModel };
export const MANAGED_COMPLEXITY_ROUTER_KEYS = new Set([
  "tiers",
  "enable_non_reasoning_tier",
  "tier_definitions",
  "fallback_tier",
  "tier_model_configs",
  "default_model",
  "plan_mode_min_tier",
  "tier_labels",
  "classifier_type",
  "capability_classifier_config",
  "llm_v2_config",
  "classifier_llm_config",
  "jev_classifier_config",
  "classifier_context_window_size",
  "classifier_context_budget_chars",
  "classifier_context_include_assistant_turns",
  "classifier_fallback",
  "classification_prompt",
  "classification_examples",
  "heuristic_first_max_tier",
  "hybrid_boundary_margin",
  "heuristic_v2_success_threshold",
  "classification_mode",
  "session_affinity",
  "session_affinity_ttl_seconds",
  "modality_routing",
  "modality_pin_override",
  "deployment_affinity",
  "adaptive",
  "adaptive_weights",
  "tier_distance_penalty",
  "adaptive_eligible",
  "return_raw_model_name",
  "tier_boundaries",
  "token_thresholds",
  "dimension_weights",
  "custom_dimensions",
  "reasoning_override_min_score",
  "enable_context_window_escalation",
  "context_window_escalation_buffer",
  "stall_escalation_enabled",
  "stall_escalation_window",
  "stall_escalation_repeat_threshold",
  "code_keywords",
  "reasoning_keywords",
  "technical_keywords",
  "simple_keywords",
  "plan_mode_patterns",
  "route_housekeeping_to_cheapest_tier",
  "housekeeping_patterns",
  "reminder_markers",
  "max_tokens_from_tier_model",
  "classifier_plugin_timeout_ms",
]);

// Managed only when the caller passes the corresponding state. A caller that does not render
// these controls must carry the stored values through untouched instead of dropping them.
const KEYWORD_MATCHING_KEYS = new Set([
  "keyword_tier_rules",
  "escalation_keywords",
  "semantic_keyword_matching",
  "embedding_model",
  "match_threshold",
]);

const UNEDITED_STORED_VALUE_KEYS: readonly (keyof ComplexityRouterConfigValue)[] = [
  "route_housekeeping_to_cheapest_tier",
  "reminder_markers",
  "max_tokens_from_tier_model",
];

const toRecord = (value: unknown): Record<string, unknown> => {
  const parsed: unknown = typeof value === "string" ? JSON.parse(value) : value;
  return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
    ? (parsed as Record<string, unknown>)
    : {};
};

export interface KeywordMatchingState {
  keywordTierRules: KeywordTierRule[];
  escalationKeywords: string[];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
  matchThreshold: number;
}

// A custom save drops the stored keys an edited tier set forbids. classification_prompt needs no
// entry here: it is a managed key, so every save rewrites it from form state and the builder
// re-emits it on both branches only when the form still holds one.
const customTierDroppedKeys = (value: ComplexityRouterConfigValue): readonly string[] =>
  value.custom_tier_set ? CUSTOM_TIER_OMITTED_KEYS : [];

export const buildUpdatedComplexityRouterConfig = (
  storedConfig: unknown,
  value: ComplexityRouterConfigValue,
  customTechnicalKeywords?: string[],
  keywordMatching?: KeywordMatchingState,
): Record<string, unknown> => {
  const stored = toRecord(storedConfig);
  const hydratedFromStored = hydrateComplexityRouterConfig(stored as StoredComplexityRouterConfig, undefined);
  const unedited = new Set<string>(
    UNEDITED_STORED_VALUE_KEYS.filter(
      (key) => key in stored && JSON.stringify(value[key]) === JSON.stringify(hydratedFromStored[key]),
    ),
  );
  const isManaged = (key: string): boolean => {
    if (unedited.has(key)) return false;
    if (key === "classifier_context_per_turn_chars") {
      return !usesClassifierContext(effectiveClassifierType(value)) || Object.prototype.hasOwnProperty.call(value, key);
    }
    if (MANAGED_COMPLEXITY_ROUTER_KEYS.has(key)) return true;
    if (key === "escalation_keywords" && isForecastClassifier(effectiveClassifierType(value))) return true;
    if (keywordMatching !== undefined && KEYWORD_MATCHING_KEYS.has(key)) return true;
    return customTechnicalKeywords !== undefined && key === "custom_technical_keywords";
  };
  const dropped = customTierDroppedKeys(value);
  const preservedConfig = Object.fromEntries(
    Object.entries(stored).filter(([key]) => !isManaged(key) && !dropped.includes(key)),
  );

  const builderParams: BuildComplexityRouterConfigParams = {
    ...builderParamsFromValue(value),
    customTechnicalKeywords: customTechnicalKeywords ?? [],
    keywordTierRules: keywordMatching?.keywordTierRules ?? [],
    semanticMatchingEnabled: keywordMatching?.semanticMatchingEnabled ?? false,
    embeddingModel: keywordMatching?.embeddingModel,
    matchThreshold: keywordMatching?.matchThreshold ?? DEFAULT_MATCH_THRESHOLD,
    escalationKeywords: keywordMatching?.escalationKeywords ?? [],
  };
  const built = buildComplexityRouterConfig(builderParams);

  // Keys this call does not own stay as the stored config left them.
  const unowned: readonly string[] = [
    ...(keywordMatching === undefined ? [...KEYWORD_MATCHING_KEYS].filter((key) => !isManaged(key)) : []),
    ...(customTechnicalKeywords === undefined ? ["custom_technical_keywords"] : []),
    ...unedited,
  ];
  return {
    ...preservedConfig,
    ...Object.fromEntries(Object.entries(built).filter(([key]) => !unowned.includes(key))),
  };
};

const EditAutoRouterModal: React.FC<EditAutoRouterModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  modelData,
  accessToken,
  userRole,
  isMemberManaged = false,
}) => {
  const [loading, setLoading] = useState(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [showValidationErrors, setShowValidationErrors] = useState<boolean>(false);
  const [editingTiers, setEditingTiers] = useState(false);
  const [routerConfig, setRouterConfig] = useState<RouterConfig | null>(null);
  const [customTechnicalKeywords, setCustomTechnicalKeywords] = useState<string[]>([]);
  const [keywordTierRules, setKeywordTierRules] = useState<KeywordTierRule[]>([]);
  const [escalationKeywords, setEscalationKeywords] = useState<string[]>([]);
  const [semanticMatchingEnabled, setSemanticMatchingEnabled] = useState<boolean>(false);
  const [embeddingModel, setEmbeddingModel] = useState<string | undefined>(undefined);
  const [matchThreshold, setMatchThreshold] = useState<number>(DEFAULT_MATCH_THRESHOLD);
  const [autoRouterCompression, setAutoRouterCompression] = useState<AutoRouterCompressionState>(
    DEFAULT_AUTO_ROUTER_COMPRESSION,
  );
  const [complexityRouterConfig, setComplexityRouterConfig] = useState<ComplexityRouterConfigValue>({
    tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    classifier_type: "heuristic",
  });
  const isComplexityRouterModel = isComplexityRouter(modelData?.litellm_params);

  const routerAvailability = useAutoRouterAvailability(
    accessToken,
    {
      saved_model_id: modelData?.model_info?.id,
      team_id: modelData?.model_info?.team_id,
      complexity_router_config: buildUpdatedComplexityRouterConfig(
        modelData?.litellm_params?.complexity_router_config,
        complexityRouterConfig,
        customTechnicalKeywords,
        { keywordTierRules, escalationKeywords, semanticMatchingEnabled, embeddingModel, matchThreshold },
      ),
    },
    isVisible && isComplexityRouterModel,
  );
  const schema = useMemo(
    () => (isComplexityRouterModel ? complexityRouterSchema : semanticRouterSchema),
    [isComplexityRouterModel],
  );
  const form = useZodForm(schema, { defaultValues: EMPTY_FORM_VALUES });

  // Mirrors the create form: the button says why it is unavailable and disables on the same
  // answer. Tiers use this modal's own rule, which allows a partly filled router, so an edit that
  // is legal today stays legal.
  const configBlockedReason = !isComplexityRouterModel
    ? null
    : (complexityRouterConfig.custom_tier_set
        ? getCustomTierRowsError(complexityRouterConfig.custom_tier_set) ??
          getMissingTiersError(activeTierRows(complexityRouterConfig))
        : (Object.values(complexityRouterConfig.tiers).every((models) => models.length === 0)
            ? "Please select at least one model for a complexity tier"
            : null) ?? getTierLabelsError(complexityRouterConfig.tier_labels)) ??
      getPlanModeTierError(complexityRouterConfig.plan_mode_min_tier, activeTierRows(complexityRouterConfig)) ??
      getKeywordTierRulesError(keywordTierRules, activeTierRows(complexityRouterConfig)) ??
      getClassifierModelError(complexityRouterConfig) ??
      getHeuristicV2SuccessThresholdError(complexityRouterConfig.heuristic_v2_success_threshold) ??
      getForecastConfigError(complexityRouterConfig) ??
      (heuristicScoringRole(complexityRouterConfig) === "decides"
        ? customDimensionsError(complexityRouterConfig.custom_dimensions)
        : null);

  const submitBlockedReason = configBlockedReason ?? routerAvailability.saveBlockedReason;

  useEffect(() => {
    if (isVisible && modelData) {
      initializeForm();
    }
  }, [isVisible, modelData]);

  useEffect(() => {
    let active = true;
    const fetchModelAccessGroups = async () => {
      if (!accessToken) return;
      try {
        const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
        setModelAccessGroups(response["data"].map((model: any) => model["id"]));
      } catch (error) {
        console.error("Error fetching model access groups:", error);
      }
    };

    const loadModels = async () => {
      if (!accessToken) return;
      setModelInfo([]);
      try {
        const uniqueModels = isMemberManaged
          ? await fetchAutoRouterModels(accessToken, modelData?.model_info?.team_id)
          : await fetchAvailableModels(accessToken);
        if (active) setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    if (isVisible) {
      fetchModelAccessGroups();
      loadModels();
    }
    return () => {
      active = false;
    };
  }, [isVisible, accessToken, isMemberManaged, modelData?.model_info?.team_id]);

  const initializeForm = () => {
    setEditingTiers(false);
    try {
      if (isComplexityRouterModel) {
        // Parse the complexity_router_config if it exists and is a string
        let parsedConfig = modelData.litellm_params?.complexity_router_config || {};
        if (typeof parsedConfig === "string") {
          parsedConfig = JSON.parse(parsedConfig);
        }

        const hydratedComplexityRouterConfig = hydrateComplexityRouterConfig(
          parsedConfig,
          modelData.litellm_params?.complexity_router_default_model,
        );
        setComplexityRouterConfig(hydratedComplexityRouterConfig);
        setCustomTechnicalKeywords(
          Array.isArray(parsedConfig.custom_technical_keywords) ? parsedConfig.custom_technical_keywords : [],
        );
        // Hydrated from the stored config, never from create-form defaults: these keys are now
        // rewritten on save, so seeding a default here would inject it into a config that never
        // had it.
        setKeywordTierRules(hydrateKeywordTierRules(parsedConfig.keyword_tier_rules));
        setEscalationKeywords(
          Array.isArray(parsedConfig.escalation_keywords)
            ? parsedConfig.escalation_keywords.filter((k: unknown): k is string => typeof k === "string")
            : [],
        );
        setSemanticMatchingEnabled(parsedConfig.semantic_keyword_matching === true);
        setEmbeddingModel(typeof parsedConfig.embedding_model === "string" ? parsedConfig.embedding_model : undefined);
        setMatchThreshold(
          typeof parsedConfig.match_threshold === "number" ? parsedConfig.match_threshold : DEFAULT_MATCH_THRESHOLD,
        );
        setAutoRouterCompression(
          hydrateAutoRouterCompression({
            auto_router_routing_compression: modelData.litellm_params?.auto_router_routing_compression,
            auto_router_model_compression: modelData.litellm_params?.auto_router_model_compression,
          }),
        );

        form.reset({
          ...EMPTY_FORM_VALUES,
          auto_router_name: modelData.model_name,
          model_access_group: modelData.model_info?.access_groups || [],
        });
        return;
      }

      // Parse the auto_router_config if it exists and is a string
      let parsedConfig = null;
      if (modelData.litellm_params?.auto_router_config) {
        if (typeof modelData.litellm_params.auto_router_config === "string") {
          parsedConfig = JSON.parse(modelData.litellm_params.auto_router_config);
        } else {
          parsedConfig = modelData.litellm_params.auto_router_config;
        }
      }

      setRouterConfig(parsedConfig);

      // Set form values
      const routerFormValues = {
        auto_router_name: modelData.model_name,
        auto_router_default_model: modelData.litellm_params?.auto_router_default_model || null,
        auto_router_embedding_model: modelData.litellm_params?.auto_router_embedding_model || null,
        model_access_group: modelData.model_info?.access_groups || [],
      };
      form.reset(routerFormValues);
    } catch (error) {
      console.error("Error parsing auto router config:", error);
      toast.fromError("Error loading auto router configuration");
    }
  };

  const saveValues = async (values: EditAutoRouterFormValues) => {
    if (routerAvailability.saveBlockedReason) {
      toast.fromError(routerAvailability.saveBlockedReason);
      return;
    }
    if (isComplexityRouterModel) {
      const { tiers, custom_tier_set, classifier_llm_config } = complexityRouterConfig;
      const rows = activeTierRows(complexityRouterConfig);
      const builtInTiersEmpty = Object.values(tiers).every((models) => models.length === 0);
      const tierSetError = custom_tier_set
        ? getCustomTierRowsError(custom_tier_set) ?? getMissingTiersError(rows)
        : builtInTiersEmpty && "Please select at least one model for a complexity tier";
      if (tierSetError) {
        setShowValidationErrors(true);
        toast.fromError(tierSetError);
        return;
      }
      const classifierError =
        getClassifierModelError(complexityRouterConfig) ??
        getHeuristicV2SuccessThresholdError(complexityRouterConfig.heuristic_v2_success_threshold) ??
        getReminderMarkersError(complexityRouterConfig.reminder_markers) ??
        getClassifierPluginTimeoutError(
          complexityRouterConfig.classifier_type,
          complexityRouterConfig.classifier_plugin_timeout_ms,
        ) ??
        getForecastConfigError(complexityRouterConfig) ??
        (heuristicScoringRole(complexityRouterConfig) === "decides"
          ? customDimensionsError(complexityRouterConfig.custom_dimensions)
          : null);
      if (classifierError) {
        setShowValidationErrors(true);
        toast.fromError(classifierError);
        return;
      }
      const classifierEffortError = getClassifierReasoningEffortError(complexityRouterConfig, modelInfo);
      if (classifierEffortError) {
        setShowValidationErrors(true);
        toast.fromError(classifierEffortError);
        return;
      }
      // Same guards the create form applies (add_auto_router_tab.tsx). The backend rejects a
      // keyword rule with no keyword, and semantic_keyword_matching without an embedding model
      // or keyword rules (complexity_router/config.py), so without these a save fails as a raw
      // 400 instead of an inline message.
      const keywordRulesError = getKeywordTierRulesError(keywordTierRules, rows);
      if (keywordRulesError) {
        setShowValidationErrors(true);
        toast.fromError(keywordRulesError);
        return;
      }

      const semanticError = getSemanticConfigError({ semanticMatchingEnabled, embeddingModel, keywordTierRules });
      if (semanticError) {
        setShowValidationErrors(true);
        toast.fromError(semanticError);
        return;
      }

      // Unlike the create form, this modal only requires one non-empty tier, so a router can reach
      // here with nothing the backend would pick as a default (see getMissingTiersError in
      // build_complexity_router_config.ts for why create never can). init_complexity_router_deployment
      // raises in that case (litellm/router.py), so block it rather than saving a router that
      // fails at init.
      const defaultModel = resolveComplexityDefaultModel(complexityRouterConfig, complexityRouterConfig.default_model);
      if (!defaultModel) {
        setShowValidationErrors(true);
        toast.fromError(
          "Add a model to the Simple or Medium tier, or pin a default model, so requests have somewhere to route.",
        );
        return;
      }

      // Dual write: complexity_router_config.default_model (the pin marker hydratePinnedDefaultModel
      // reads back) and complexity_router_default_model (what the backend routes on) must always be
      // written together from the same value. Same pairing in add_auto_router_tab.tsx.
      const keywordMatching = {
        keywordTierRules,
        escalationKeywords,
        semanticMatchingEnabled,
        embeddingModel,
        matchThreshold,
      };
      const updatedConfig = buildUpdatedComplexityRouterConfig(
        modelData.litellm_params?.complexity_router_config,
        complexityRouterConfig,
        customTechnicalKeywords,
        keywordMatching,
      );
      const serverVerdict = await validateAutoRouterConfig(accessToken, updatedConfig, modelData?.model_info?.team_id);
      const dryRunError = dryRunRejection(serverVerdict);
      if (dryRunError) {
        setShowValidationErrors(true);
        toast.fromError(dryRunError);
        return;
      }

      const updatedLitellmParams = {
        ...modelData.litellm_params,
        complexity_router_config: updatedConfig,
        complexity_router_default_model: defaultModel,
        ...(isMemberManaged
          ? {}
          : buildAutoRouterCompressionPatch(autoRouterCompression, modelData.litellm_params ?? {})),
      };
      const updatedModelInfo = {
        ...modelData.model_info,
        access_groups: values.model_access_group || [],
      };

      await modelPatchUpdateCall(
        accessToken,
        isMemberManaged
          ? {
              litellm_params: {
                complexity_router_config: updatedConfig,
                complexity_router_default_model: defaultModel,
              },
            }
          : { model_name: values.auto_router_name, litellm_params: updatedLitellmParams, model_info: updatedModelInfo },
        modelData.model_info.id,
      );

      toast.success("Auto router configuration updated successfully");
      const updatedModelData = {
        ...modelData,
        model_name: values.auto_router_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };
      onSuccess(updatedModelData);
      onCancel();
      return;
    }

    // Prepare the updated litellm_params
    const updatedLitellmParams = {
      ...modelData.litellm_params,
      auto_router_config: serializeRouterConfig(routerConfig),
      auto_router_default_model: values.auto_router_default_model,
      auto_router_embedding_model: values.auto_router_embedding_model || undefined,
    };

    // Prepare updated model_info
    const updatedModelInfo = {
      ...modelData.model_info,
      access_groups: values.model_access_group || [],
    };

    const updateData = {
      model_name: values.auto_router_name,
      litellm_params: updatedLitellmParams,
      model_info: updatedModelInfo,
    };

    await modelPatchUpdateCall(accessToken, updateData, modelData.model_info.id);

    const updatedModelData = {
      ...modelData,
      model_name: values.auto_router_name,
      litellm_params: updatedLitellmParams,
      model_info: updatedModelInfo,
    };

    toast.success("Auto router configuration updated successfully");
    onSuccess(updatedModelData);
    onCancel();
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      await form.handleSubmit(saveValues, () => {
        toast.fromError("Failed to update auto router configuration");
      })();
    } catch (error) {
      console.error("Error updating auto router:", error);
      toast.fromError(error);
    } finally {
      setLoading(false);
    }
  };

  const modelChoices: ModelChoice[] = [
    ...modelInfo.map((model) => ({ value: model.model_group, label: model.model_group })),
    { value: "custom", label: "Enter custom model name" },
  ];

  const routerNameField = (
    <FormField control={form.control} name="auto_router_name" label="Auto Router Name">
      {({ ref, ...field }) => (
        <Input {...field} ref={ref} readOnly={isMemberManaged} placeholder="e.g., auto_router_1, smart_routing" />
      )}
    </FormField>
  );

  return (
    <AutoRouterAvailabilityContext.Provider value={routerAvailability}>
      <Dialog open={isVisible} onOpenChange={(open) => !open && onCancel()}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-4xl">
          <TooltipProvider>
            <DialogHeader>
              <DialogTitle>Edit Auto Router Configuration</DialogTitle>
              <DialogDescription>
                Edit the auto router configuration including routing logic, default models, and access settings.
              </DialogDescription>
            </DialogHeader>

            <form onSubmit={(event) => event.preventDefault()} noValidate>
              <FieldGroup>
                {routerNameField}

                {isComplexityRouterModel ? (
                  /* Complexity Router Configuration */
                  <div className="w-full">
                    <AutoRouterClassifierTabs value={complexityRouterConfig} onChange={setComplexityRouterConfig}>
                      <ComplexityRouterConfig
                        editingTiers={editingTiers}
                        onEditingTiersChange={setEditingTiers}
                        showValidationErrors={showValidationErrors}
                        modelInfo={modelInfo}
                        value={complexityRouterConfig}
                        onChange={(config) => {
                          setComplexityRouterConfig(config);
                        }}
                        customTechnicalKeywords={customTechnicalKeywords}
                        onCustomTechnicalKeywordsChange={setCustomTechnicalKeywords}
                        keywordTierRules={keywordTierRules}
                        onKeywordTierRulesChange={setKeywordTierRules}
                        keywordRulesError={getKeywordTierRulesError(
                          keywordTierRules,
                          activeTierRows(complexityRouterConfig),
                        )}
                        semanticMatchingEnabled={semanticMatchingEnabled}
                        onSemanticMatchingEnabledChange={setSemanticMatchingEnabled}
                        embeddingModel={embeddingModel}
                        onEmbeddingModelChange={setEmbeddingModel}
                        matchThreshold={matchThreshold}
                        onMatchThresholdChange={setMatchThreshold}
                        escalationKeywords={escalationKeywords}
                        onEscalationKeywordsChange={setEscalationKeywords}
                        autoRouterCompression={autoRouterCompression}
                        onAutoRouterCompressionChange={isMemberManaged ? undefined : setAutoRouterCompression}
                      />
                    </AutoRouterClassifierTabs>
                  </div>
                ) : (
                  <>
                    {/* Router Configuration Builder */}
                    <div className="w-full">
                      <RouterConfigBuilder
                        modelInfo={modelInfo}
                        value={routerConfig}
                        onChange={(config) => {
                          setRouterConfig(config);
                        }}
                      />
                    </div>

                    <FormField control={form.control} name="auto_router_default_model" label="Default Model">
                      {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                        <ModelChoiceCombobox
                          id={id}
                          value={value}
                          onChange={onChange}
                          choices={modelChoices}
                          placeholder="Select a default model"
                          ariaInvalid={ariaInvalid}
                          ariaDescribedBy={ariaDescribedBy}
                        />
                      )}
                    </FormField>

                    <FormField control={form.control} name="auto_router_embedding_model" label="Embedding Model">
                      {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                        <ModelChoiceCombobox
                          id={id}
                          value={value}
                          onChange={onChange}
                          choices={modelChoices}
                          placeholder="Select an embedding model"
                          ariaInvalid={ariaInvalid}
                          ariaDescribedBy={ariaDescribedBy}
                        />
                      )}
                    </FormField>
                  </>
                )}

                {userRole === "Admin" && !isMemberManaged && (
                  <FormField
                    control={form.control}
                    name="model_access_group"
                    label={labelWithHint("Model Access Groups", "Control who can access this auto router")}
                  >
                    {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                      <AccessGroupTagsCombobox
                        id={id}
                        value={value}
                        onChange={onChange}
                        options={modelAccessGroups}
                        ariaInvalid={ariaInvalid}
                        ariaDescribedBy={ariaDescribedBy}
                      />
                    )}
                  </FormField>
                )}
              </FieldGroup>
            </form>

            <DialogFooter>
              <Button variant="outline" onClick={onCancel}>
                Cancel
              </Button>
              {submitBlockedReason === null ? (
                <Button disabled={loading} onClick={handleSubmit}>
                  {loading && <UiLoadingSpinner className="size-4" />}
                  Save Changes
                </Button>
              ) : (
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <Button disabled onClick={handleSubmit}>
                        Save Changes
                      </Button>
                    }
                  />
                  <TooltipContent>{submitBlockedReason}</TooltipContent>
                </Tooltip>
              )}
            </DialogFooter>
          </TooltipProvider>
        </DialogContent>
      </Dialog>
    </AutoRouterAvailabilityContext.Provider>
  );
};

export default EditAutoRouterModal;
