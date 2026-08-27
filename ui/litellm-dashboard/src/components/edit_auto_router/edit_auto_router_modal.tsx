import React, { useEffect, useMemo, useState } from "react";
import { z } from "zod/v4";
import { toast } from "@/lib/toast";
import { CircleHelp } from "lucide-react";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import AccessGroupTagsCombobox from "../add_model/AccessGroupTagsCombobox";
import ModelChoiceCombobox, { type ModelChoice } from "../add_model/ModelChoiceCombobox";
import { modelAvailableCall, modelPatchUpdateCall } from "../networking";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import RouterConfigBuilder from "../add_model/RouterConfigBuilder";
import { hydrateTierModelParams, normalizeTierModels } from "../add_model/complexity_router_tiers";
import { type ActiveTierSet, activeTierRows, resolveComplexityDefaultModel } from "../add_model/tier_rows";
import { isComplexityRouter } from "../add_model/auto_router_strategies";
import {
  type BuildComplexityRouterConfigParams,
  buildComplexityRouterConfig,
  getClassifierModelError,
  getKeywordTierRulesError,
  getSemanticConfigError,
  getPlanModeTierError,
  getTierLabelsError,
  hydrateTierLabels,
} from "../add_model/build_complexity_router_config";
import { KeywordTierRule } from "../add_model/KeywordTierRules";
import { DEFAULT_MATCH_THRESHOLD } from "../add_model/SemanticKeywordMatching";
import { hydrateKeywordTierRules } from "../add_model/complexity_router_keywords";
import {
  hydrateDimensionWeights,
  hydrateReasoningOverrideMinScore,
  hydrateTierBoundaries,
  hydrateTokenThresholds,
} from "../add_model/heuristic_scoring_knobs";
import ComplexityRouterConfig, {
  AdaptiveEligible,
  AdaptiveRouterWeights,
  ClassifierLLMConfig,
  ClassifierType,
  ComplexityRouterConfigValue,
  ComplexityTiers,
  DEFAULT_ADAPTIVE_WEIGHTS,
  DEFAULT_SESSION_AFFINITY,
  DEFAULT_DEPLOYMENT_AFFINITY,
  DEFAULT_TIER_DISTANCE_PENALTY,
} from "../add_model/ComplexityRouterConfig";
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
}

// Keys this modal rewrites from its own form state on save. Anything absent from this set is
// carried through untouched from the stored config, so a key only belongs here once the modal
// actually renders a control that can set it.
/** The complexity_router_config as it comes back from the proxy, before any hydration. Fields the
 * hydrators validate themselves stay `unknown`; the ones assigned straight through carry their type. */
export interface StoredComplexityRouterConfig {
  tiers?: Partial<Record<keyof ComplexityTiers, unknown>>;
  tier_model_configs?: unknown;
  default_model?: string | null;
  plan_mode_min_tier?: unknown;
  heuristic_first_max_tier?: unknown;
  tier_labels?: unknown;
  classifier_type?: ClassifierType;
  classifier_llm_config?: ClassifierLLMConfig;
  classifier_context_window_size?: unknown;
  classifier_context_budget_chars?: unknown;
  classifier_context_include_assistant_turns?: unknown;
  classifier_fallback?: unknown;
  tier_boundaries?: unknown;
  token_thresholds?: unknown;
  dimension_weights?: unknown;
  reasoning_override_min_score?: unknown;
  session_affinity?: unknown;
  deployment_affinity?: unknown;
  adaptive?: boolean;
  adaptive_weights?: AdaptiveRouterWeights;
  tier_distance_penalty?: number;
  adaptive_eligible?: AdaptiveEligible;
  return_raw_model_name?: boolean;
}

/**
 * The stored complexity_router_config as form state. Every key in MANAGED_COMPLEXITY_ROUTER_KEYS is
 * rewritten from this state on save, so a key missing here is silently dropped from the saved config.
 */
export const hydrateComplexityRouterConfig = (
  parsedConfig: StoredComplexityRouterConfig,
  complexityRouterDefaultModel: string | null | undefined,
): ComplexityRouterConfigValue => {
  const hydratedTiers: ComplexityTiers = {
    SIMPLE: normalizeTierModels(parsedConfig.tiers?.SIMPLE),
    MEDIUM: normalizeTierModels(parsedConfig.tiers?.MEDIUM),
    COMPLEX: normalizeTierModels(parsedConfig.tiers?.COMPLEX),
    REASONING: normalizeTierModels(parsedConfig.tiers?.REASONING),
  };

  return {
    tiers: hydratedTiers,
    tier_model_params: hydrateTierModelParams(parsedConfig.tiers, parsedConfig.tier_model_configs),
    default_model: hydratePinnedDefaultModel(parsedConfig.default_model, complexityRouterDefaultModel, {
      tiers: hydratedTiers,
    }),
    plan_mode_min_tier:
      typeof parsedConfig.plan_mode_min_tier === "string" && parsedConfig.plan_mode_min_tier.trim() !== ""
        ? parsedConfig.plan_mode_min_tier
        : undefined,
    tier_labels: hydrateTierLabels(parsedConfig.tier_labels),
    classifier_type: parsedConfig.classifier_type || "heuristic",
    classifier_llm_config: parsedConfig.classifier_llm_config,
    classifier_context_window_size:
      typeof parsedConfig.classifier_context_window_size === "number"
        ? parsedConfig.classifier_context_window_size
        : undefined,
    classifier_context_budget_chars:
      typeof parsedConfig.classifier_context_budget_chars === "number"
        ? parsedConfig.classifier_context_budget_chars
        : undefined,
    classifier_context_include_assistant_turns:
      typeof parsedConfig.classifier_context_include_assistant_turns === "boolean"
        ? parsedConfig.classifier_context_include_assistant_turns
        : undefined,
    classifier_fallback:
      parsedConfig.classifier_fallback === "default_model" || parsedConfig.classifier_fallback === "heuristic"
        ? parsedConfig.classifier_fallback
        : undefined,
    heuristic_first_max_tier:
      typeof parsedConfig.heuristic_first_max_tier === "string" && parsedConfig.heuristic_first_max_tier.trim() !== ""
        ? parsedConfig.heuristic_first_max_tier
        : undefined,
    tier_boundaries: hydrateTierBoundaries(parsedConfig.tier_boundaries),
    token_thresholds: hydrateTokenThresholds(parsedConfig.token_thresholds),
    dimension_weights: hydrateDimensionWeights(parsedConfig.dimension_weights),
    reasoning_override_min_score: hydrateReasoningOverrideMinScore(parsedConfig.reasoning_override_min_score),
    session_affinity:
      typeof parsedConfig.session_affinity === "boolean" ? parsedConfig.session_affinity : DEFAULT_SESSION_AFFINITY,
    deployment_affinity:
      typeof parsedConfig.deployment_affinity === "boolean"
        ? parsedConfig.deployment_affinity
        : DEFAULT_DEPLOYMENT_AFFINITY,
    adaptive: parsedConfig.adaptive || false,
    adaptive_weights: parsedConfig.adaptive_weights,
    tier_distance_penalty: parsedConfig.tier_distance_penalty,
    adaptive_eligible: parsedConfig.adaptive_eligible || "all",
    return_raw_model_name: parsedConfig.return_raw_model_name || false,
  };
};

export const MANAGED_COMPLEXITY_ROUTER_KEYS = new Set([
  "tiers",
  "tier_model_configs",
  "default_model",
  "plan_mode_min_tier",
  "tier_labels",
  "classifier_type",
  "classifier_llm_config",
  "classifier_context_window_size",
  "classifier_context_budget_chars",
  "classifier_context_include_assistant_turns",
  "classifier_fallback",
  "heuristic_first_max_tier",
  "session_affinity",
  "deployment_affinity",
  "adaptive",
  "adaptive_weights",
  "tier_distance_penalty",
  "adaptive_eligible",
  "return_raw_model_name",
  "tier_boundaries",
  "token_thresholds",
  "dimension_weights",
  "reasoning_override_min_score",
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

const toRecord = (value: unknown): Record<string, unknown> => {
  const parsed: unknown = typeof value === "string" ? JSON.parse(value) : value;
  return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
    ? (parsed as Record<string, unknown>)
    : {};
};

// A pin lives in two places: complexity_router_config.default_model (this UI's own marker, added
// by PR #36615) and litellm_params.complexity_router_default_model (what the backend reads). Only
// the marker proves an operator picked it, because before #36615 every save wrote a tier-derived
// value into litellm_params. So with no marker, a litellm_params value counts as a pin only when
// it diverges from what the tiers alone derive; a match stays unpinned and keeps tracking tiers.
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

export interface KeywordMatchingState {
  keywordTierRules: KeywordTierRule[];
  escalationKeywords: string[];
  semanticMatchingEnabled: boolean;
  embeddingModel: string | undefined;
  matchThreshold: number;
}

export const buildUpdatedComplexityRouterConfig = (
  storedConfig: unknown,
  value: ComplexityRouterConfigValue,
  customTechnicalKeywords?: string[],
  keywordMatching?: KeywordMatchingState,
): Record<string, unknown> => {
  const isManaged = (key: string): boolean => {
    if (MANAGED_COMPLEXITY_ROUTER_KEYS.has(key)) return true;
    if (keywordMatching !== undefined && KEYWORD_MATCHING_KEYS.has(key)) return true;
    return customTechnicalKeywords !== undefined && key === "custom_technical_keywords";
  };
  const preservedConfig = Object.fromEntries(Object.entries(toRecord(storedConfig)).filter(([key]) => !isManaged(key)));

  const builderParams: BuildComplexityRouterConfigParams = {
    tiers: value.tiers,
    defaultModel: value.default_model,
    planModeMinTier: value.plan_mode_min_tier,
    heuristicFirstMaxTier: value.heuristic_first_max_tier,
    tierLabels: value.tier_labels,
    classifierType: value.classifier_type,
    classifierLlmConfig: value.classifier_llm_config,
    classifierContextWindowSize: value.classifier_context_window_size,
    classifierContextBudgetChars: value.classifier_context_budget_chars,
    classifierContextIncludeAssistantTurns: value.classifier_context_include_assistant_turns,
    classifierFallback: value.classifier_fallback,
    sessionAffinity: value.session_affinity ?? DEFAULT_SESSION_AFFINITY,
    deploymentAffinity: value.deployment_affinity ?? DEFAULT_DEPLOYMENT_AFFINITY,
    customTechnicalKeywords: customTechnicalKeywords ?? [],
    keywordTierRules: keywordMatching?.keywordTierRules ?? [],
    semanticMatchingEnabled: keywordMatching?.semanticMatchingEnabled ?? false,
    embeddingModel: keywordMatching?.embeddingModel,
    matchThreshold: keywordMatching?.matchThreshold ?? DEFAULT_MATCH_THRESHOLD,
    escalationKeywords: keywordMatching?.escalationKeywords ?? [],
    adaptive: value.adaptive ?? false,
    adaptiveWeights: value.adaptive_weights ?? DEFAULT_ADAPTIVE_WEIGHTS,
    tierDistancePenalty: value.tier_distance_penalty ?? DEFAULT_TIER_DISTANCE_PENALTY,
    adaptiveEligible: value.adaptive_eligible ?? "all",
    returnRawModelName: value.return_raw_model_name ?? false,
    tierBoundaries: value.tier_boundaries,
    tokenThresholds: value.token_thresholds,
    dimensionWeights: value.dimension_weights,
    reasoningOverrideMinScore: value.reasoning_override_min_score,
    tierModelParams: value.tier_model_params,
  };
  const built = buildComplexityRouterConfig(builderParams);

  // Keys this call does not own stay as the stored config left them.
  const unowned: readonly string[] = [
    ...(keywordMatching === undefined ? KEYWORD_MATCHING_KEYS : []),
    ...(customTechnicalKeywords === undefined ? ["custom_technical_keywords"] : []),
  ];
  return {
    ...preservedConfig,
    ...Object.fromEntries(Object.entries(built).filter(([key]) => !unowned.includes(key))),
  };
};

const sharedShape = {
  auto_router_name: z.string().min(1, "Auto router name is required"),
  model_access_group: z.array(z.string()),
};

const complexityRouterShape = {
  ...sharedShape,
  auto_router_default_model: z.string(),
  auto_router_embedding_model: z.string(),
};

const semanticRouterShape = {
  ...sharedShape,
  auto_router_default_model: z.string().min(1, "Default model is required"),
  auto_router_embedding_model: z.string().min(1, "Embedding model is required"),
};

const complexityRouterSchema = z.object(complexityRouterShape);
const semanticRouterSchema = z.object(semanticRouterShape);

type EditAutoRouterFormValues = z.infer<typeof semanticRouterSchema>;

const EMPTY_FORM_VALUES: EditAutoRouterFormValues = {
  auto_router_name: "",
  auto_router_default_model: "",
  auto_router_embedding_model: "",
  model_access_group: [],
};

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const EditAutoRouterModal: React.FC<EditAutoRouterModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  modelData,
  accessToken,
  userRole,
}) => {
  const [loading, setLoading] = useState(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [showValidationErrors, setShowValidationErrors] = useState<boolean>(false);
  const [routerConfig, setRouterConfig] = useState<any>(null);
  const [customTechnicalKeywords, setCustomTechnicalKeywords] = useState<string[]>([]);
  const [keywordTierRules, setKeywordTierRules] = useState<KeywordTierRule[]>([]);
  const [escalationKeywords, setEscalationKeywords] = useState<string[]>([]);
  const [semanticMatchingEnabled, setSemanticMatchingEnabled] = useState<boolean>(false);
  const [embeddingModel, setEmbeddingModel] = useState<string | undefined>(undefined);
  const [matchThreshold, setMatchThreshold] = useState<number>(DEFAULT_MATCH_THRESHOLD);
  const [complexityRouterConfig, setComplexityRouterConfig] = useState<ComplexityRouterConfigValue>({
    tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    classifier_type: "heuristic",
  });
  const isComplexityRouterModel = isComplexityRouter(modelData?.litellm_params);

  const schema = useMemo(
    () => (isComplexityRouterModel ? complexityRouterSchema : semanticRouterSchema),
    [isComplexityRouterModel],
  );
  const form = useZodForm(schema, { defaultValues: EMPTY_FORM_VALUES });

  // Mirrors the create form: the button says why it is unavailable and disables on the same
  // answer. Tiers use this modal's own rule, which allows a partly filled router, so an edit that
  // is legal today stays legal.
  const submitBlockedReason = !isComplexityRouterModel
    ? null
    : (Object.values(complexityRouterConfig.tiers).every((models) => models.length === 0)
        ? "Please select at least one model for a complexity tier"
        : null) ??
      getTierLabelsError(complexityRouterConfig.tier_labels) ??
      getPlanModeTierError(complexityRouterConfig.plan_mode_min_tier, activeTierRows(complexityRouterConfig)) ??
      getKeywordTierRulesError(keywordTierRules, activeTierRows(complexityRouterConfig)) ??
      getClassifierModelError(complexityRouterConfig);

  useEffect(() => {
    if (isVisible && modelData) {
      initializeForm();
    }
  }, [isVisible, modelData]);

  useEffect(() => {
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
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    if (isVisible) {
      fetchModelAccessGroups();
      loadModels();
    }
  }, [isVisible, accessToken]);

  const initializeForm = () => {
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
      form.reset({
        auto_router_name: modelData.model_name,
        auto_router_default_model: modelData.litellm_params?.auto_router_default_model || "",
        auto_router_embedding_model: modelData.litellm_params?.auto_router_embedding_model || "",
        model_access_group: modelData.model_info?.access_groups || [],
      });
    } catch (error) {
      console.error("Error parsing auto router config:", error);
      toast.fromError("Error loading auto router configuration");
    }
  };

  const saveValues = async (values: EditAutoRouterFormValues) => {
    if (isComplexityRouterModel) {
      const { tiers, classifier_type, classifier_llm_config } = complexityRouterConfig;
      if (Object.values(tiers).every((models) => models.length === 0)) {
        setShowValidationErrors(true);
        toast.fromError("Please select at least one model for a complexity tier");
        return;
      }
      const classifierError = getClassifierModelError(complexityRouterConfig);
      if (classifierError) {
        setShowValidationErrors(true);
        toast.fromError(classifierError);
        return;
      }
      // Same guards the create form applies (add_auto_router_tab.tsx). The backend rejects a
      // keyword rule with no keyword, and semantic_keyword_matching without an embedding model
      // or keyword rules (complexity_router/config.py), so without these a save fails as a raw
      // 400 instead of an inline message.
      const keywordRulesError = getKeywordTierRulesError(keywordTierRules, activeTierRows(complexityRouterConfig));
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
      const updatedLitellmParams = {
        ...modelData.litellm_params,
        complexity_router_config: buildUpdatedComplexityRouterConfig(
          modelData.litellm_params?.complexity_router_config,
          complexityRouterConfig,
          customTechnicalKeywords,
          {
            keywordTierRules,
            escalationKeywords,
            semanticMatchingEnabled,
            embeddingModel,
            matchThreshold,
          },
        ),
        complexity_router_default_model: defaultModel,
      };
      const updatedModelInfo = {
        ...modelData.model_info,
        access_groups: values.model_access_group || [],
      };

      await modelPatchUpdateCall(
        accessToken,
        { model_name: values.auto_router_name, litellm_params: updatedLitellmParams, model_info: updatedModelInfo },
        modelData.model_info.id,
      );

      toast.success("Auto router configuration updated successfully");
      onSuccess({
        ...modelData,
        model_name: values.auto_router_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      });
      onCancel();
      return;
    }

    // Prepare the updated litellm_params
    const updatedLitellmParams = {
      ...modelData.litellm_params,
      auto_router_config: JSON.stringify(routerConfig),
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
      toast.fromError("Failed to update auto router configuration");
    } finally {
      setLoading(false);
    }
  };

  const modelChoices: ModelChoice[] = [
    ...modelInfo.map((model) => ({ value: model.model_group, label: model.model_group })),
    { value: "custom", label: "Enter custom model name" },
  ];

  return (
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
              <FormField control={form.control} name="auto_router_name" label="Auto Router Name">
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="e.g., auto_router_1, smart_routing" />}
              </FormField>

              {isComplexityRouterModel ? (
                /* Complexity Router Configuration */
                <div className="w-full">
                  <ComplexityRouterConfig
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
                    semanticMatchingEnabled={semanticMatchingEnabled}
                    onSemanticMatchingEnabledChange={setSemanticMatchingEnabled}
                    embeddingModel={embeddingModel}
                    onEmbeddingModelChange={setEmbeddingModel}
                    matchThreshold={matchThreshold}
                    onMatchThresholdChange={setMatchThreshold}
                    escalationKeywords={escalationKeywords}
                    onEscalationKeywordsChange={setEscalationKeywords}
                  />
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

              {userRole === "Admin" && (
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
  );
};

export default EditAutoRouterModal;
