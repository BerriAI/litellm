import { AutoRouterAvailabilityContext, useAutoRouterAvailability } from "./AutoRouterAvailability";
import { decisionDeploymentName } from "./jev_classifier_config";
import DecisionModelSelect from "./DecisionModelSelect";
import ClassifierPrimarySettings from "./ClassifierPrimarySettings";
import AutoRouterClassifierTabs from "./AutoRouterClassifierTabs";
import { getForecastConfigError, isForecastClassifier } from "../add_model/forecast_classifier_config";
import React, { useEffect, useState } from "react";
import BlockedReasonTooltip from "./BlockedReasonTooltip";
import { useSwitchyardOption } from "./SwitchyardRouterForm";
import { useTranslation } from "react-i18next";
import { useQuery } from "@tanstack/react-query";
import { useWatch } from "react-hook-form";
import { z } from "zod";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import AutoRouterPresetSelect, { type PresetAvailability } from "./AutoRouterPresetSelect";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import AccessGroupTagsCombobox from "./AccessGroupTagsCombobox";
import { getRouterSettingsCall, modelAvailableCall, validateAutoRouterConfig, type Team } from "../networking";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { all_admin_roles, isProxyAdminRole } from "@/utils/roles";
import { canCreateAutoRouterForTeam, canModifyModel, type ModelWriteScope } from "@/utils/modelPermissions";
import TeamDropdown from "../common_components/team_dropdown";
import { type AddAutoRouterValues, handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { fetchAutoRouterModels, fetchAvailableModels, type ModelGroup } from "@/components/llm_calls/fetch_models";
import { autoRouterListKey, fetchAllModelDeployments } from "@/app/(dashboard)/hooks/models/useModels";
import ComplexityRouterConfig, {
  ComplexityRouterConfigValue,
  effectiveClassifierType,
  usesLlmClassifier,
  heuristicScoringRole,
} from "./ComplexityRouterConfig";
import { KeywordTierRule } from "./KeywordTierRules";
import { customDimensionsError } from "./custom_dimensions";
import { DEFAULT_ESCALATION_KEYWORDS } from "./EscalationKeywords";
import {
  type AutoRouterCompressionState,
  buildAutoRouterCompressionParams,
  DEFAULT_AUTO_ROUTER_COMPRESSION,
} from "./buildAutoRouterCompression";
import { DEFAULT_MATCH_THRESHOLD } from "./SemanticKeywordMatching";
import {
  BuildComplexityRouterConfigParams,
  buildComplexityRouterConfig,
  getKeywordTierRulesError,
  getClassifierModelError,
  getHeuristicV2SuccessThresholdError,
  getReminderMarkersError,
  getClassifierPluginTimeoutError,
  getClassifierReasoningEffortError,
  getMissingTiersError,
  getPlanModeTierError,
  getSemanticConfigError,
  getTierLabelsError,
  dryRunRejection,
} from "./build_complexity_router_config";
import { builderParamsFromValue } from "./complexity_router_builder_params";
import { activeTierName, activeTierRows, getCustomTierRowsError, resolveComplexityDefaultModel } from "./tier_rows";
import { buildAutoRouterTestTargets, AutoRouterTestTarget } from "./build_auto_router_test_targets";
import { AutoRouterConnectionTestDialog } from "./auto_router_connection_test";
import {
  buildAutoRouterRoutingTestRequest,
  JEV_CONNECTION_TEST_PROMPT,
} from "./build_auto_router_routing_test_request";
import AutoRouterRoutingTest from "./AutoRouterRoutingTest";
import { toast } from "@/lib/toast";
import i18n from "@/i18n/i18n";
import {
  getMissingModelsInPreset,
  getReferencedModelsError,
  buildEmptyPrefill,
  buildPresetPrefill,
  buildModelAvailability,
  deploymentRefsFromModelInfo,
  modelGroupAliasTargets,
  ModelAvailability,
  PresetPrefill,
  AutoRouterPreset,
} from "@/lib/autorouter_presets";
import { useAutoRouterPresets } from "@/app/(dashboard)/hooks/autoRouter/useAutoRouterPresets";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import AutoRouterSimpleSetup from "./AutoRouterSimpleSetup";
import {
  deployedRouterModels,
  nextRouterName,
  simpleSetupError,
  recommendSolverPair,
  solverPairConfig,
  type RouterGoal,
  type FamilyPreference,
} from "./auto_setup";

interface AddAutoRouterTabProps {
  handleOk: () => void;
  accessToken: string;
  userRole: string;
  userId?: string | null;
  /**
   * How this caller must scope what they create. A team admin has to name a team, because
   * POST /model/new rejects an unscoped create from any non-proxy-admin; without the selector
   * their submit is a guaranteed 403.
   */
  createScope?: ModelWriteScope;
  teams?: Team[] | null;
}

const NO_PRESETS: AutoRouterPreset[] = [];
const NO_ALIASES: ReadonlyMap<string, string> = new Map();

// Why the submit is unavailable, or null when it is available. The button reads this to disable
// itself and to say what is missing, so the two can never give different answers. Checks the
// config actually being built, not which preset (if any) it came from: a preset only ever
// prefills once (handlePresetChange), and everything after that is edited exactly like Custom.
export const getSubmitBlockedReason = (
  config: ComplexityRouterConfigValue,
  keywordTierRules: KeywordTierRule[],
  referencedModelsParams: Parameters<typeof getReferencedModelsError>[0],
  ...capabilities: [availability: ModelAvailability, modelInfo?: readonly ModelGroup[]]
): string | null => {
  const [availability, modelInfo = []] = capabilities;
  return (
    (config.custom_tier_set
      ? getCustomTierRowsError(config.custom_tier_set)
      : getTierLabelsError(config.tier_labels)) ??
    (isForecastClassifier(config.classifier_type)
      ? getForecastConfigError(config)
      : getMissingTiersError(activeTierRows(config))) ??
    getPlanModeTierError(config.plan_mode_min_tier, activeTierRows(config)) ??
    getKeywordTierRulesError(keywordTierRules, activeTierRows(config)) ??
    (effectiveClassifierType(config) === "jev" && !config.jev_classifier_config?.deployment_name
      ? i18n.t("addModel.addAutoRouterTab.decisionModelRequired")
      : null) ??
    getClassifierModelError(config) ??
    getHeuristicV2SuccessThresholdError(config.heuristic_v2_success_threshold) ??
    getReminderMarkersError(config.reminder_markers) ??
    getClassifierPluginTimeoutError(config.classifier_type, config.classifier_plugin_timeout_ms) ??
    (heuristicScoringRole(config) === "decides" ? customDimensionsError(config.custom_dimensions) : null) ??
    getClassifierReasoningEffortError(config, modelInfo) ??
    getReferencedModelsError(referencedModelsParams, availability)
  );
};

const autoRouterSchema = (requiresTeamScope: boolean) =>
  z.object({
    auto_router_name: z.string().min(1, { error: () => i18n.t("addModel.addAutoRouterTab.nameRequired") }),
    team_id: z
      .string()
      .nullable()
      .refine((teamId) => !requiresTeamScope || Boolean(teamId), {
        error: () => i18n.t("addModel.addAutoRouterTab.teamRequired"),
      }),
    model_access_group: z.array(z.string()).optional(),
  });

type AddAutoRouterFormValues = z.infer<ReturnType<typeof autoRouterSchema>>;

const EMPTY_FORM_VALUES: AddAutoRouterFormValues = {
  auto_router_name: "auto-router",
  team_id: null,
  model_access_group: undefined,
};

const teamScopePayload = (requiresTeamScope: boolean, teamId: string | null): { team_id?: string } =>
  requiresTeamScope && teamId ? { team_id: teamId } : {};

const AddAutoRouterTab: React.FC<AddAutoRouterTabProps> = ({
  handleOk,
  accessToken,
  userRole,
  userId,
  createScope = "unscoped-ok",
  teams = null,
}) => {
  const { t } = useTranslation();
  const requiresTeamScope = createScope === "team-required";
  const form = useZodForm(autoRouterSchema(requiresTeamScope), { defaultValues: EMPTY_FORM_VALUES });
  const watchedName = useWatch({ control: form.control, name: "auto_router_name" });
  const switchyard = useSwitchyardOption(accessToken, watchedName, isProxyAdminRole(userRole));
  const watchedTeamId = useWatch({ control: form.control, name: "team_id" });
  const actor = { userRole, userID: userId ?? null, isViewOnly: false };
  const isMemberManaged =
    requiresTeamScope &&
    !canModifyModel(actor, teams, {
      teamId: watchedTeamId,
      isDbModel: true,
    });
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);

  const [advancedConfig, setComplexityRouterConfig] = useState<ComplexityRouterConfigValue>({
    tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    classifier_type: "heuristic",
    classification_mode: "user_turn",
  });

  const [advanced, setAdvanced] = useState(false);
  const [prefillAttempted, setPrefillAttempted] = useState(false);
  const [goal, setGoal] = useState<RouterGoal>("balanced");
  const [family, setFamily] = useState<FamilyPreference>("auto");
  const [solverOverrides, setSolverOverrides] = useState<{ everyday?: string; hard?: string }>({});

  const [customTechnicalKeywords, setCustomTechnicalKeywords] = useState<string[]>([]);
  const [keywordTierRules, setKeywordTierRules] = useState<KeywordTierRule[]>([]);
  const [semanticMatchingEnabled, setSemanticMatchingEnabled] = useState<boolean>(false);
  const [embeddingModel, setEmbeddingModel] = useState<string | undefined>(undefined);
  const [matchThreshold, setMatchThreshold] = useState<number>(DEFAULT_MATCH_THRESHOLD);
  const [escalationKeywords, setEscalationKeywords] = useState<string[]>(DEFAULT_ESCALATION_KEYWORDS);
  const [autoRouterCompression, setAutoRouterCompression] = useState<AutoRouterCompressionState>(
    DEFAULT_AUTO_ROUTER_COMPRESSION,
  );
  const [showValidationErrors, setShowValidationErrors] = useState<boolean>(false);
  const [editingTiers, setEditingTiers] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const [selectedPreset, setSelectedPreset] = useState<string | undefined>(undefined);

  const [isRoutingTestVisible, setIsRoutingTestVisible] = useState<boolean>(false);
  const [isTestModalVisible, setIsTestModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  const [connectionTestId, setConnectionTestId] = useState<number>(0);
  const [testTargets, setTestTargets] = useState<AutoRouterTestTarget[]>([]);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
      setModelAccessGroups(response["data"].map((model: any) => model["id"]));
    };
    fetchModelAccessGroups();
  }, [accessToken]);

  const {
    data,
    isLoading: groupsLoading,
    isError: modelsError,
    refetch: refetchModels,
  } = useQuery({
    queryKey: ["availableModels", "autoRouter", accessToken, ...(requiresTeamScope ? [watchedTeamId] : [])],
    queryFn: () =>
      requiresTeamScope ? fetchAutoRouterModels(accessToken, watchedTeamId) : fetchAvailableModels(accessToken),
    enabled: Boolean(accessToken && (!requiresTeamScope || watchedTeamId)),
  });
  const {
    data: deployments,
    isLoading: deploymentsLoading,
    isError: deploymentsError,
    refetch: refetchDeployments,
  } = useQuery({
    queryKey: autoRouterListKey(userId ?? "", userRole),
    queryFn: () => fetchAllModelDeployments(accessToken, userId ?? "", userRole),
    enabled: Boolean(accessToken),
  });
  const { data: aliasTargets = NO_ALIASES, isLoading: aliasesLoading } = useQuery({
    queryKey: ["routerSettings", "modelGroupAlias", accessToken],
    queryFn: async () =>
      modelGroupAliasTargets((await getRouterSettingsCall(accessToken))?.current_values?.model_group_alias),
    enabled: Boolean(accessToken) && isProxyAdminRole(userRole),
  });
  const modelsLoading = groupsLoading || deploymentsLoading || aliasesLoading;
  useEffect(() => {
    if (!deployments || form.getFieldState("auto_router_name").isDirty) return;
    form.setValue(
      "auto_router_name",
      nextRouterName(deployments.flatMap((row) => (row.model_name ? [row.model_name] : []))),
    );
  }, [deployments, form]);

  const modelInfo = React.useMemo(() => data ?? [], [data]);
  const {
    data: presetsData,
    isPending: presetsPending,
    isError: presetsError,
    refetch: refetchPresets,
  } = useAutoRouterPresets();
  const presets = presetsData ?? NO_PRESETS;
  const presetsUnavailable = presetsError && presetsData === undefined;
  // react-query keeps the last successful list around when a later refetch fails, so isError alone
  // can't tell "never loaded" apart from "loaded, then a background refetch errored" - only the
  // former leaves us with nothing trustworthy to verify a preset's models against.
  const modelsUnverifiable = modelsError && data === undefined;
  const inventoryUnavailable = modelsUnverifiable || (deploymentsError && deployments === undefined);

  const isAdmin = all_admin_roles.includes(userRole);

  const availability = React.useMemo(
    () =>
      buildModelAvailability(
        modelInfo.map((m) => m.model_group),
        deploymentRefsFromModelInfo(deployments ?? []),
        new Map(modelInfo.map((m) => [m.model_group, m.providers ?? []])),
        aliasTargets,
      ),
    [modelInfo, deployments, aliasTargets],
  );
  const groupsOnlyAvailability = React.useMemo(
    () =>
      buildModelAvailability(
        modelInfo.map((m) => m.model_group),
        [],
      ),
    [modelInfo],
  );
  const solverModels = React.useMemo(
    () => deployedRouterModels(modelInfo, deployments ?? [], aliasTargets),
    [modelInfo, deployments, aliasTargets],
  );
  const recommended = React.useMemo(
    () => recommendSolverPair(solverModels, goal, family),
    [solverModels, goal, family],
  );
  const everyday = solverOverrides.everyday
    ? solverModels.find((model) => model.name === solverOverrides.everyday)
    : recommended?.everyday;
  const hard = solverOverrides.hard
    ? solverModels.find((model) => model.name === solverOverrides.hard)
    : recommended?.hard;
  const simpleConfig = everyday && hard ? solverPairConfig(everyday.name, hard.name) : null;
  const complexityRouterConfig = advanced
    ? advancedConfig
    : simpleConfig ?? {
        classifier_type: "heuristic" as const,
        classification_mode: "user_turn" as const,
        tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
      };
  const openAdvanced = () => {
    setComplexityRouterConfig(complexityRouterConfig);
    setPrefillAttempted(simpleConfig !== null);
    setAdvanced(true);
  };

  // Withhold preset selection until its models are verified against a successfully loaded list.
  // Reuse this check for application and disabled state to prevent load races.
  const presetAvailability = React.useCallback(
    (preset: AutoRouterPreset): PresetAvailability => {
      if (modelsLoading) return { kind: "loading" };
      if (modelsUnverifiable) return { kind: "unverifiable" };
      const missing = getMissingModelsInPreset(preset, availability);
      if (missing.length > 0) return { kind: "missing_models", models: missing };
      return {
        kind: "available",
        viaDeployments: getMissingModelsInPreset(preset, groupsOnlyAvailability).length > 0,
      };
    },
    [modelsLoading, modelsUnverifiable, availability, groupsOnlyAvailability],
  );

  const applyPrefill = (prefill: PresetPrefill) => {
    setEditingTiers(false);
    setComplexityRouterConfig(prefill.complexityRouterConfig);
    setCustomTechnicalKeywords(prefill.customTechnicalKeywords);
    setKeywordTierRules(prefill.keywordTierRules);
    setSemanticMatchingEnabled(prefill.semanticMatchingEnabled);
    setEmbeddingModel(prefill.embeddingModel);
    setMatchThreshold(prefill.matchThreshold);
    setEscalationKeywords(prefill.escalationKeywords);
  };

  if (advanced && !prefillAttempted && !modelsLoading && simpleConfig) {
    setPrefillAttempted(true);
    setComplexityRouterConfig((current) =>
      !current.custom_tier_set &&
      !isForecastClassifier(current.classifier_type) &&
      activeTierRows(current).every((row) => row.models.length === 0)
        ? { ...current, tiers: simpleConfig.tiers }
        : current,
    );
  }

  const handlePresetChange = (presetKey: string | undefined) => {
    setPrefillAttempted(true);
    if (!presetKey || presetKey === "custom") {
      setSelectedPreset(presetKey);
      applyPrefill(buildEmptyPrefill());

      return;
    }

    const preset = presets.find((p) => p.key === presetKey);
    // Refuse to apply a preset whose models are not verified available. The dropdown disables
    // these options, so this is a guard against a stale click resolving after the list changed.
    if (!preset) return;
    const presetState = presetAvailability(preset);
    if (presetState.kind !== "available") return;

    setSelectedPreset(presetKey);
    applyPrefill(buildPresetPrefill(preset.complexity_router_config, availability));
  };

  const referencedModelsParams = {
    tiers: Object.fromEntries(activeTierRows(complexityRouterConfig).map((row) => [activeTierName(row), row.models])),
    classifierType: effectiveClassifierType(complexityRouterConfig),
    classifierLlmConfig: complexityRouterConfig.classifier_llm_config,
    decisionDeploymentName: decisionDeploymentName(complexityRouterConfig.jev_classifier_config),
    semanticMatchingEnabled,
    embeddingModel,
    defaultModel: complexityRouterConfig.default_model,
  };

  const submitBlockedReason = getSubmitBlockedReason(
    complexityRouterConfig,
    keywordTierRules,
    referencedModelsParams,
    groupsOnlyAvailability,
    modelInfo,
  );

  const complexityRouterConfigParams: BuildComplexityRouterConfigParams = {
    ...builderParamsFromValue(complexityRouterConfig),
    customTechnicalKeywords,
    keywordTierRules,
    semanticMatchingEnabled,
    embeddingModel,
    matchThreshold,
    escalationKeywords,
  };
  const routerAvailability = useAutoRouterAvailability(
    accessToken,
    {
      team_id: requiresTeamScope ? watchedTeamId : undefined,
      complexity_router_config:
        submitBlockedReason === null ? { ...buildComplexityRouterConfig(complexityRouterConfigParams) } : null,
    },
    !requiresTeamScope || Boolean(watchedTeamId),
  );
  const simpleBlockedReason = advanced ? null : simpleSetupError(modelsLoading, inventoryUnavailable, everyday, hard);
  const saveBlockedReason =
    !advanced && requiresTeamScope && !watchedTeamId
      ? t("addModel.addAutoRouterTab.selectTeamToSeeModels")
      : simpleBlockedReason ?? submitBlockedReason ?? routerAvailability.saveBlockedReason;
  const handleClassifierChange = (config: ComplexityRouterConfigValue) => {
    switchyard.deselect();
    if (advanced || effectiveClassifierType(config) === effectiveClassifierType(complexityRouterConfig)) {
      setSelectedPreset(undefined);
      setComplexityRouterConfig(config);
      return;
    }
    setComplexityRouterConfig(config);
    setPrefillAttempted(simpleConfig !== null);
    setAdvanced(true);
  };
  const jevRequestParams =
    effectiveClassifierType(complexityRouterConfig) === "jev"
      ? {
          prompt: JEV_CONNECTION_TEST_PROMPT,
          config: buildComplexityRouterConfig(complexityRouterConfigParams),
          defaultModel: resolveComplexityDefaultModel(complexityRouterConfig, complexityRouterConfig.default_model),
          routerName: watchedName,
          teamId: requiresTeamScope ? watchedTeamId ?? undefined : undefined,
        }
      : undefined;
  const jevRequest = jevRequestParams ? buildAutoRouterRoutingTestRequest(jevRequestParams) : undefined;

  const submitRecommendedRouter = async (name: string) => {
    // The one answer the submit button reads, so a disabled button and a refused submit cannot
    // disagree about why. The handler needs it in its own right: the form fires this on Enter
    // regardless of the button's disabled state.
    const blockedReason =
      saveBlockedReason ??
      getSubmitBlockedReason(
        complexityRouterConfig,
        keywordTierRules,
        referencedModelsParams,
        groupsOnlyAvailability,
        modelInfo,
      ) ??
      getSemanticConfigError({ semanticMatchingEnabled, embeddingModel, keywordTierRules });
    if (blockedReason) {
      setShowValidationErrors(true);
      toast.fromError(blockedReason);
      return;
    }

    const defaultModel = resolveComplexityDefaultModel(complexityRouterConfig, complexityRouterConfig.default_model);
    const validatedFields = requiresTeamScope
      ? (["auto_router_name", "team_id"] as const)
      : (["auto_router_name"] as const);

    if (!(await form.trigger(validatedFields))) {
      toast.fromError(t("addModel.addAutoRouterTab.requiredFieldsMissing"));
      return;
    }

    // auto_router_default_model (-> litellm_params, read by the backend at init) and
    // complexity_router_config.default_model (-> the pin marker read back on edit, see
    // hydratePinnedDefaultModel in edit_auto_router_modal.tsx) must both come from the same
    // `defaultModel`, or the two fields diverge and hydration's divergence check misfires.
    const complexityRouterConfigPayload = buildComplexityRouterConfig(complexityRouterConfigParams);
    const serverVerdict = await validateAutoRouterConfig(
      accessToken,
      complexityRouterConfigPayload as unknown as Record<string, unknown>,
      requiresTeamScope ? form.getValues("team_id") ?? undefined : undefined,
    );
    const dryRunError = dryRunRejection(serverVerdict);
    if (dryRunError) {
      setShowValidationErrors(true);
      toast.fromError(dryRunError);
      return;
    }

    const submitValues: AddAutoRouterValues = {
      auto_router_name: name,
      ...teamScopePayload(requiresTeamScope, form.getValues("team_id")),
      auto_router_default_model: defaultModel,
      model_type: "complexity_router",
      complexity_router_config: complexityRouterConfigPayload,
      ...(isMemberManaged
        ? {}
        : {
            model_access_group: form.getValues("model_access_group"),
            ...buildAutoRouterCompressionParams(autoRouterCompression),
          }),
    };

    await handleAddAutoRouterSubmit(submitValues, accessToken, () => form.reset(EMPTY_FORM_VALUES), handleOk);
  };

  const submitSwitchyard = async (name: string) => {
    if (requiresTeamScope && !(await form.trigger(["auto_router_name", "team_id"]))) {
      toast.fromError(t("addModel.addAutoRouterTab.requiredFieldsMissing"));
      return;
    }
    const accessGroups = isMemberManaged ? undefined : form.getValues("model_access_group");
    await switchyard.submit(
      name,
      {
        ...teamScopePayload(requiresTeamScope, form.getValues("team_id")),
        ...(accessGroups?.length ? { access_groups: accessGroups } : {}),
      },
      () => form.reset(EMPTY_FORM_VALUES),
      handleOk,
    );
  };

  const handleAutoRouterSubmit = async () => {
    if (isSubmitting) return;
    const name = form.getValues("auto_router_name");
    if (!name) {
      setShowValidationErrors(true);
      void form.trigger("auto_router_name");
      toast.fromError(t("addModel.addAutoRouterTab.enterRouterName"));
      return;
    }

    setIsSubmitting(true);
    try {
      await (switchyard.selected ? submitSwitchyard(name) : submitRecommendedRouter(name));
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleTestConnection = () => {
    const testTargetParams = {
      tiers: activeTierRows(complexityRouterConfig).map(
        (row) => [activeTierName(row), row.models] as [string, string[]],
      ),
      semanticMatchingEnabled,
      embeddingModel,
      defaultModel: resolveComplexityDefaultModel(complexityRouterConfig, complexityRouterConfig.default_model),
      classifier: usesLlmClassifier(effectiveClassifierType(complexityRouterConfig))
        ? {
            model: complexityRouterConfig.classifier_llm_config?.model ?? "",
            reasoningEffort: complexityRouterConfig.classifier_llm_config?.reasoning_effort,
          }
        : undefined,
    };
    const targets = buildAutoRouterTestTargets(testTargetParams, t);

    if (targets.length === 0) {
      toast.fromError(t("addModel.addAutoRouterTab.selectTierModel"));
      return;
    }

    setTestTargets(targets);
    setConnectionTestId((id) => id + 1);
    setIsTestingConnection(true);
    setIsTestModalVisible(true);
  };

  const configurationForm = (
    <ComplexityRouterConfig
      creationFlow
      tierSetup={
        <AutoRouterPresetSelect
          selectedPreset={selectedPreset}
          presets={presets}
          presetAvailability={presetAvailability}
          onPresetChange={handlePresetChange}
          pending={presetsPending}
          unavailable={presetsUnavailable}
          onRetry={() => void refetchPresets()}
        />
      }
      editingTiers={editingTiers}
      onEditingTiersChange={setEditingTiers}
      modelInfo={modelInfo}
      value={complexityRouterConfig}
      onChange={setComplexityRouterConfig}
      customTechnicalKeywords={customTechnicalKeywords}
      onCustomTechnicalKeywordsChange={setCustomTechnicalKeywords}
      keywordTierRules={keywordTierRules}
      onKeywordTierRulesChange={setKeywordTierRules}
      keywordRulesError={getKeywordTierRulesError(keywordTierRules, activeTierRows(complexityRouterConfig))}
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
      showValidationErrors={showValidationErrors}
    />
  );

  return (
    <AutoRouterAvailabilityContext.Provider value={routerAvailability}>
      <TooltipProvider>
        <Card>
          <CardContent>
            <form onSubmit={form.handleSubmit(() => handleAutoRouterSubmit())} noValidate>
              <div className="mb-4">
                <FormField
                  control={form.control}
                  name="auto_router_name"
                  label={labelWithHint(
                    t("addModel.addAutoRouterTab.routerName"),
                    t("addModel.addAutoRouterTab.routerNameHint"),
                  )}
                >
                  {({ ref, ...field }) => (
                    <Input {...field} ref={ref} placeholder={t("addModel.addAutoRouterTab.routerNamePlaceholder")} />
                  )}
                </FormField>
              </div>
              <div className={advanced ? "mb-6 border-b pb-6" : ""}>
                {isAdmin && advanced && (
                  <div>
                    <FormField
                      control={form.control}
                      name="model_access_group"
                      label={labelWithHint(
                        t("addModel.addAutoRouterTab.modelAccessGroup"),
                        t("addModel.addAutoRouterTab.modelAccessGroupHint"),
                      )}
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
                  </div>
                )}
              </div>
              {requiresTeamScope && (
                <FormField
                  control={form.control}
                  name="team_id"
                  label={labelWithHint(
                    t("addModel.addAutoRouterTab.selectTeam"),
                    t("addModel.addAutoRouterTab.selectTeamHint"),
                  )}
                >
                  {({ id, value, onChange }) => (
                    <TeamDropdown
                      id={id}
                      value={value}
                      onChange={onChange}
                      filterTeam={(team) => canCreateAutoRouterForTeam(actor, team)}
                    />
                  )}
                </FormField>
              )}

              <AutoRouterClassifierTabs
                switchyard={switchyard.option({ capable: hard?.name, efficient: everyday?.name })}
                value={complexityRouterConfig}
                onChange={handleClassifierChange}
                decisionModel={
                  <DecisionModelSelect
                    value={complexityRouterConfig}
                    onChange={setComplexityRouterConfig}
                    modelInfo={modelInfo}
                    deployments={deployments}
                  />
                }
                judgeModel={
                  <ClassifierPrimarySettings
                    section="judge"
                    value={complexityRouterConfig}
                    onChange={setComplexityRouterConfig}
                    modelOptions={modelInfo
                      .filter((model) => model.mode !== "embedding")
                      .map((model) => ({ value: model.model_group, label: model.model_group }))}
                    showValidationErrors={showValidationErrors}
                  />
                }
              >
                {advanced ? (
                  <FieldGroup>
                    {modelsUnverifiable && (
                      <p className="text-sm text-destructive">
                        {t("addModel.addAutoRouterTab.modelsLoadFailed")}{" "}
                        <button type="button" className="underline" onClick={() => void refetchModels()}>
                          {t("addModel.addAutoRouterTab.retry")}
                        </button>
                      </p>
                    )}
                    {configurationForm}
                  </FieldGroup>
                ) : (
                  <AutoRouterSimpleSetup
                    customized={Boolean(solverOverrides.everyday || solverOverrides.hard)}
                    hasSuggestions={Boolean(recommended || recommendSolverPair(solverModels, goal, "auto"))}
                    onReset={() => setSolverOverrides({})}
                    goal={goal}
                    family={family}
                    models={solverModels}
                    everyday={everyday}
                    hard={hard}
                    waitingForTeam={requiresTeamScope && !watchedTeamId}
                    loading={modelsLoading}
                    error={inventoryUnavailable}
                    notice={
                      Object.keys(solverOverrides).length > 0
                        ? t("addModel.addAutoRouterTab.choicesKept")
                        : recommended?.notice ?? null
                    }
                    onGoalChange={setGoal}
                    onFamilyChange={(next) => {
                      setFamily(next);
                      setSolverOverrides({});
                    }}
                    onModelChange={(role, model) =>
                      setSolverOverrides({ everyday: everyday?.name, hard: hard?.name, [role]: model })
                    }
                    onAdvanced={openAdvanced}
                    onRetry={() => {
                      void refetchModels();
                      void refetchDeployments();
                    }}
                  />
                )}
              </AutoRouterClassifierTabs>
              <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <a
                        href="https://github.com/BerriAI/litellm/issues"
                        className="text-sm text-primary underline-offset-4 hover:underline"
                      >
                        {t("addModel.addAutoRouterTab.needHelp")}
                      </a>
                    }
                  />
                  <TooltipContent>{t("addModel.addAutoRouterTab.helpTooltip")}</TooltipContent>
                </Tooltip>
                <div className="flex flex-wrap gap-2">
                  <BlockedReasonTooltip reason={submitBlockedReason}>
                    <Button
                      type="button"
                      variant="outline"
                      data-testid="auto-router-test-routing-btn"
                      hidden={switchyard.selected}
                      disabled={submitBlockedReason !== null || isSubmitting}
                      onClick={() => setIsRoutingTestVisible(true)}
                    >
                      {t("addModel.addAutoRouterTab.testRouting")}
                    </Button>
                  </BlockedReasonTooltip>
                  <Button
                    type="button"
                    variant="outline"
                    data-testid="auto-router-test-connect-btn"
                    hidden={switchyard.selected}
                    onClick={handleTestConnection}
                    disabled={isTestingConnection}
                  >
                    {isTestingConnection && <UiLoadingSpinner className="size-4" />}
                    {t("addModel.addAutoRouterTab.testConnection")}
                  </Button>
                  <BlockedReasonTooltip reason={switchyard.selected ? null : saveBlockedReason}>
                    <Button
                      type="button"
                      disabled={(!switchyard.selected && saveBlockedReason !== null) || isSubmitting}
                      onClick={() => {
                        void handleAutoRouterSubmit();
                      }}
                    >
                      {t("addModel.addAutoRouterTab.addAutoRouter")}
                    </Button>
                  </BlockedReasonTooltip>
                </div>
              </div>
            </form>
          </CardContent>
        </Card>

        <Dialog open={isRoutingTestVisible} onOpenChange={(open) => !open && setIsRoutingTestVisible(false)}>
          <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[760px]">
            <DialogHeader>
              <DialogTitle>{t("addModel.addAutoRouterTab.testRoutingTitle")}</DialogTitle>
            </DialogHeader>
            {isRoutingTestVisible && (
              <AutoRouterRoutingTest
                accessToken={accessToken}
                config={buildComplexityRouterConfig(complexityRouterConfigParams)}
                defaultModel={resolveComplexityDefaultModel(
                  complexityRouterConfig,
                  complexityRouterConfig.default_model,
                )}
                routerName={watchedName}
                teamId={requiresTeamScope ? watchedTeamId ?? undefined : undefined}
              />
            )}
            <DialogFooter>
              {" "}
              <Button variant="outline" onClick={() => setIsRoutingTestVisible(false)}>
                {t("addModel.addAutoRouterTab.close")}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <AutoRouterConnectionTestDialog
          open={isTestModalVisible}
          onClose={() => {
            setIsTestModalVisible(false);
            setIsTestingConnection(false);
          }}
          testId={connectionTestId}
          accessToken={accessToken}
          targets={testTargets}
          jevRequest={jevRequest}
          onTestComplete={() => setIsTestingConnection(false)}
        />
      </TooltipProvider>
    </AutoRouterAvailabilityContext.Provider>
  );
};

export default AddAutoRouterTab;
