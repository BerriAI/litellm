import { AutoRouterAvailabilityContext, useAutoRouterAvailability } from "./AutoRouterAvailability";
import AutoRouterClassifierTabs from "./AutoRouterClassifierTabs";
import { getForecastConfigError, isForecastClassifier } from "../add_model/forecast_classifier_config";
import React, { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useWatch } from "react-hook-form";
import { z } from "zod/v4";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import AccessGroupTagsCombobox from "./AccessGroupTagsCombobox";
import { modelAvailableCall, validateAutoRouterConfig, type Team } from "../networking";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { all_admin_roles } from "@/utils/roles";
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
import { tierRowLabel } from "./complexity_router_tiers";
import { buildAutoRouterTestTargets, AutoRouterTestTarget } from "./build_auto_router_test_targets";
import { AutoRouterConnectionTestDialog } from "./auto_router_connection_test";
import {
  buildAutoRouterRoutingTestRequest,
  JEV_CONNECTION_TEST_PROMPT,
} from "./build_auto_router_routing_test_request";
import AutoRouterRoutingTest from "./AutoRouterRoutingTest";
import { toast } from "@/lib/toast";
import {
  getMissingModelsInPreset,
  getReferencedModelsError,
  buildEmptyPrefill,
  buildPresetPrefill,
  buildModelAvailability,
  deploymentRefsFromModelInfo,
  ModelAvailability,
  PresetPrefill,
  AutoRouterPreset,
} from "@/lib/autorouter_presets";
import { useAutoRouterPresets } from "@/app/(dashboard)/hooks/autoRouter/useAutoRouterPresets";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { buildAutomaticRouterConfig, buildPreferredTierModels } from "./auto_setup";

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

type PresetAvailability =
  | { kind: "available"; viaDeployments: boolean }
  | { kind: "loading" }
  | { kind: "unverifiable" }
  | { kind: "missing_models"; models: readonly string[] };

// Every non-"available" state disables the option. Selection derives from this same function
// (see presetAvailability below), so an option a caller can click is always one that can be applied.
const presetDisabledHint = (availability: PresetAvailability): string | null => {
  switch (availability.kind) {
    case "available":
      return null;
    case "loading":
      return "Checking model availability...";
    case "unverifiable":
      return "Cannot verify these models are available";
    case "missing_models":
      return `Missing: ${availability.models.join(", ")}`;
  }
};

// "loading"/"unverifiable" are transient system states, not a gap specific to this preset; only a
// caller-specific missing-model reason gets the alarming red treatment.
const isPresetHintAlarming = (availability: PresetAvailability): boolean => availability.kind === "missing_models";

const NO_PRESETS: AutoRouterPreset[] = [];

// A one-line summary of what's configured, shown when the detailed section is collapsed so a
// caller can see the shape of the config without opening it.
const tierConfigSummary = (config: ComplexityRouterConfigValue): string => {
  const parts = activeTierRows(config)
    .filter((row) => row.models.length > 0)
    .map((row) => `${tierRowLabel(row, config.tier_labels)}: ${row.models.join(", ")}`);
  return parts.length > 0 ? parts.join(" · ") : "No tiers configured yet";
};

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
    auto_router_name: z.string().min(1, "Auto router name is required"),
    team_id: z
      .string()
      .nullable()
      .refine((teamId) => !requiresTeamScope || Boolean(teamId), "Please select a team to continue"),
    model_access_group: z.array(z.string()).optional(),
  });

type AddAutoRouterFormValues = z.infer<ReturnType<typeof autoRouterSchema>>;

const EMPTY_FORM_VALUES: AddAutoRouterFormValues = {
  auto_router_name: "",
  team_id: null,
  model_access_group: undefined,
};

const teamScopePayload = (requiresTeamScope: boolean, teamId: string | null): { team_id?: string } =>
  requiresTeamScope && teamId ? { team_id: teamId } : {};

const BlockedReasonTooltip: React.FC<{ reason: string | null; children: React.ReactElement }> = ({
  reason,
  children,
}) =>
  reason === null ? (
    children
  ) : (
    <Tooltip>
      <TooltipTrigger render={children} />
      <TooltipContent>{reason}</TooltipContent>
    </Tooltip>
  );

const AddAutoRouterTab: React.FC<AddAutoRouterTabProps> = ({
  handleOk,
  accessToken,
  userRole,
  userId,
  createScope = "unscoped-ok",
  teams = null,
}) => {
  const requiresTeamScope = createScope === "team-required";
  const form = useZodForm(autoRouterSchema(requiresTeamScope), { defaultValues: EMPTY_FORM_VALUES });
  const watchedName = useWatch({ control: form.control, name: "auto_router_name" });
  const watchedTeamId = useWatch({ control: form.control, name: "team_id" });
  const actor = { userRole, userID: userId ?? null, isViewOnly: false };
  const isMemberManaged =
    requiresTeamScope &&
    !canModifyModel(actor, teams, {
      teamId: watchedTeamId,
      isDbModel: true,
    });
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);

  const [complexityRouterConfig, setComplexityRouterConfig] = useState<ComplexityRouterConfigValue>({
    tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    classifier_type: "heuristic",
  });

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
    queryKey: ["availableModels", "autoRouter", accessToken, ...(isMemberManaged ? [watchedTeamId] : [])],
    queryFn: () =>
      isMemberManaged ? fetchAutoRouterModels(accessToken, watchedTeamId) : fetchAvailableModels(accessToken),
    enabled: Boolean(accessToken && (!isMemberManaged || watchedTeamId)),
  });
  const { data: deployments, isLoading: deploymentsLoading } = useQuery({
    queryKey: autoRouterListKey(userId ?? "", userRole),
    queryFn: () => fetchAllModelDeployments(accessToken, userId ?? "", userRole),
    enabled: Boolean(accessToken),
  });
  const modelsLoading = groupsLoading || deploymentsLoading;
  const modelInfo = React.useMemo(() => data ?? [], [data]);
  const {
    data: presetsData,
    isPending: presetsPending,
    isError: presetsError,
    refetch: refetchPresets,
  } = useAutoRouterPresets();
  const presets = presetsData ?? NO_PRESETS;
  const automaticSetupLoading = modelsLoading || presetsPending;
  const presetsUnavailable = presetsError && presetsData === undefined;
  // react-query keeps the last successful list around when a later refetch fails, so isError alone
  // can't tell "never loaded" apart from "loaded, then a background refetch errored" - only the
  // former leaves us with nothing trustworthy to verify a preset's models against.
  const modelsUnverifiable = modelsError && data === undefined;

  const isAdmin = all_admin_roles.includes(userRole);

  const availability = React.useMemo(
    () =>
      buildModelAvailability(
        modelInfo.map((m) => m.model_group),
        deploymentRefsFromModelInfo(deployments ?? []),
      ),
    [modelInfo, deployments],
  );
  const groupsOnlyAvailability = React.useMemo(
    () =>
      buildModelAvailability(
        modelInfo.map((m) => m.model_group),
        [],
      ),
    [modelInfo],
  );
  const preferredTierModels = React.useMemo(
    () => buildPreferredTierModels(presets, availability),
    [presets, availability],
  );
  const automaticRouterConfig = React.useMemo(
    () => buildAutomaticRouterConfig(modelInfo, deployments ?? [], preferredTierModels),
    [modelInfo, deployments, preferredTierModels],
  );

  // A preset's models can only be trusted against a successfully loaded list. Selection and the
  // greyed-out state derive from this one function, so a preset that cannot be selected can never
  // have been applied: while loading we withhold selection rather than let a caller pick a preset
  // whose models we cannot yet verify, and a failed fetch leaves every preset unverifiable. This
  // makes the load-race (pick during loading, then discover a missing model) unrepresentable.
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

  const sortedPresetOptions = React.useMemo(
    () =>
      presets
        .map((preset) => ({ preset, availability: presetAvailability(preset) }))
        .sort((a, b) => Number(b.availability.kind === "available") - Number(a.availability.kind === "available")),
    [presets, presetAvailability],
  );

  const templateItems = React.useMemo(
    () => [
      ...sortedPresetOptions.map(({ preset }) => ({ value: preset.key, label: preset.label })),
      { value: "custom", label: "Custom Configuration" },
    ],
    [sortedPresetOptions],
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

  const handleAutomaticSetup = () => {
    if (automaticRouterConfig === null) return;
    setSelectedPreset(undefined);
    setComplexityRouterConfig({
      ...complexityRouterConfig,
      tiers: { ...complexityRouterConfig.tiers, ...automaticRouterConfig.tiers },
      tier_model_params: { ...complexityRouterConfig.tier_model_params, ...automaticRouterConfig.tier_model_params },
    });

    toast.success("Models selected", { description: tierConfigSummary(automaticRouterConfig) });
  };

  const handlePresetChange = (presetKey: string | undefined) => {
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
  const saveBlockedReason = submitBlockedReason ?? routerAvailability.saveBlockedReason;
  const handleClassifierChange = (config: ComplexityRouterConfigValue) => {
    setSelectedPreset(undefined);
    setComplexityRouterConfig(config);
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
      toast.fromError("Please fill in all required fields");
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

  const handleAutoRouterSubmit = async () => {
    if (isSubmitting) return;
    const name = form.getValues("auto_router_name");
    if (!name) {
      setShowValidationErrors(true);
      void form.trigger("auto_router_name");
      toast.fromError("Please enter an Auto Router Name");
      return;
    }

    setIsSubmitting(true);
    try {
      await submitRecommendedRouter(name);
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
    const targets = buildAutoRouterTestTargets(testTargetParams);

    if (targets.length === 0) {
      toast.fromError("Please select at least one model for a complexity tier");
      return;
    }

    setTestTargets(targets);
    setConnectionTestId((id) => id + 1);
    setIsTestingConnection(true);
    setIsTestModalVisible(true);
  };

  const configurationForm = (
    <ComplexityRouterConfig
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
  const forecast = isForecastClassifier(complexityRouterConfig.classifier_type);

  return (
    <AutoRouterAvailabilityContext.Provider value={routerAvailability}>
      <TooltipProvider>
        <Card>
          <CardContent>
            <form onSubmit={form.handleSubmit(() => handleAutoRouterSubmit())} noValidate>
              <div className="mb-6">
                <FormField
                  control={form.control}
                  name="auto_router_name"
                  label={labelWithHint("Auto Router Name", "Unique name for this auto router configuration")}
                >
                  {({ ref, ...field }) => (
                    <Input {...field} ref={ref} placeholder="e.g., smart_router, auto_router_1" />
                  )}
                </FormField>
              </div>
              {requiresTeamScope && (
                <FormField
                  control={form.control}
                  name="team_id"
                  label={labelWithHint(
                    "Select Team",
                    "Select the team this auto router belongs to. Only keys for this team will be able to call it.",
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

              <AutoRouterClassifierTabs value={complexityRouterConfig} onChange={handleClassifierChange}>
                <FieldGroup>
                  <div className="space-y-4 empty:hidden">
                    {!forecast && (
                      <>
                        {!automaticSetupLoading && automaticRouterConfig && !complexityRouterConfig.custom_tier_set && (
                          <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-border bg-muted px-4 py-3">
                            <div>
                              <p className="text-sm font-medium text-foreground">Choose models quickly</p>
                              <p className="text-sm text-muted-foreground">
                                Fill the tiers while keeping your classifier and settings.
                              </p>
                            </div>
                            <Button
                              type="button"
                              data-testid="configure-automatically-button"
                              onClick={handleAutomaticSetup}
                            >
                              Choose models for me
                            </Button>
                          </div>
                        )}

                        <div className="space-y-2">
                          <label className="block text-sm font-medium text-foreground mb-2">Template</label>
                          <Select
                            items={templateItems}
                            value={selectedPreset ?? null}
                            onValueChange={(presetKey: string | null) => handlePresetChange(presetKey ?? undefined)}
                          >
                            <SelectTrigger data-testid="template-selector" className="w-full">
                              <SelectValue placeholder="Choose a template or select Custom to define your own" />
                            </SelectTrigger>
                            <SelectContent>
                              {sortedPresetOptions.map(({ preset, availability: presetState }) => {
                                const disabledHint = presetDisabledHint(presetState);
                                const hintClass = isPresetHintAlarming(presetState)
                                  ? "text-destructive"
                                  : "text-muted-foreground";
                                const matchedHint =
                                  presetState.kind === "available" && presetState.viaDeployments
                                    ? "Matches your deployments"
                                    : null;

                                return (
                                  <SelectItem
                                    key={preset.key}
                                    value={preset.key}
                                    label={preset.label}
                                    disabled={disabledHint !== null}
                                    title={disabledHint ?? preset.description}
                                  >
                                    <div>
                                      <div className="font-medium">{preset.label}</div>
                                      <div className="text-xs text-muted-foreground">{preset.description}</div>
                                      <div className="text-xs text-muted-foreground">
                                        Replaces classifier and model choices
                                      </div>
                                      {disabledHint && (
                                        <div className={`text-xs mt-1 ${hintClass}`}>{disabledHint}</div>
                                      )}
                                      {matchedHint && <div className="text-xs mt-1 text-success">{matchedHint}</div>}
                                    </div>
                                  </SelectItem>
                                );
                              })}
                              <SelectItem value="custom" label="Custom Configuration">
                                <div>
                                  <div className="font-medium">Custom Configuration</div>
                                  <div className="text-xs text-muted-foreground">
                                    Define your auto router from scratch
                                  </div>
                                </div>
                              </SelectItem>
                            </SelectContent>
                          </Select>
                          {presetsPending && (
                            <div className="text-xs mt-1 text-muted-foreground">Loading templates...</div>
                          )}
                          {presetsUnavailable && (
                            <div className="text-xs mt-1 text-destructive">
                              Could not load templates, so only Custom Configuration is shown.{" "}
                              <button type="button" className="underline" onClick={() => void refetchPresets()}>
                                Retry
                              </button>
                            </div>
                          )}
                        </div>
                      </>
                    )}
                    {modelsUnverifiable && (
                      <div className="text-xs mt-1 text-destructive">
                        Could not load available models.{" "}
                        <button type="button" className="underline" onClick={() => refetchModels()}>
                          Retry
                        </button>
                      </div>
                    )}
                  </div>

                  {configurationForm}

                  {isAdmin && (
                    <FormField
                      control={form.control}
                      name="model_access_group"
                      label={labelWithHint(
                        "Model Access Group",
                        "Use model access groups to control who can access this auto router",
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
                  )}

                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <Tooltip>
                      <TooltipTrigger
                        render={
                          <a
                            href="https://github.com/BerriAI/litellm/issues"
                            className="text-sm text-primary underline-offset-4 hover:underline"
                          >
                            Need Help?
                          </a>
                        }
                      />
                      <TooltipContent>Get help on our github</TooltipContent>
                    </Tooltip>
                    <div className="flex flex-wrap gap-2">
                      <BlockedReasonTooltip reason={submitBlockedReason}>
                        <Button
                          type="button"
                          variant="outline"
                          data-testid="auto-router-test-routing-btn"
                          disabled={submitBlockedReason !== null || isSubmitting}
                          onClick={() => setIsRoutingTestVisible(true)}
                        >
                          Test Routing
                        </Button>
                      </BlockedReasonTooltip>
                      <Button
                        type="button"
                        variant="outline"
                        data-testid="auto-router-test-connect-btn"
                        onClick={handleTestConnection}
                        disabled={isTestingConnection}
                      >
                        {isTestingConnection && <UiLoadingSpinner className="size-4" />}
                        Test Connection
                      </Button>
                      <BlockedReasonTooltip reason={saveBlockedReason}>
                        <Button
                          type="button"
                          disabled={saveBlockedReason !== null || isSubmitting}
                          onClick={() => {
                            void handleAutoRouterSubmit();
                          }}
                        >
                          Add Auto Router
                        </Button>
                      </BlockedReasonTooltip>
                    </div>
                  </div>
                </FieldGroup>
              </AutoRouterClassifierTabs>
            </form>
          </CardContent>
        </Card>

        <Dialog open={isRoutingTestVisible} onOpenChange={(open) => !open && setIsRoutingTestVisible(false)}>
          <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[760px]">
            <DialogHeader>
              <DialogTitle>Test Routing</DialogTitle>
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
                Close
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
