import React, { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useWatch } from "react-hook-form";
import { ChevronDown, ChevronRight, CircleHelp } from "lucide-react";
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
import { modelAvailableCall } from "../networking";
import { all_admin_roles } from "@/utils/roles";
import { type ModelWriteScope } from "@/utils/modelPermissions";
import TeamDropdown from "../common_components/team_dropdown";
import { type AddAutoRouterValues, handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import { autoRouterListKey, fetchAllModelDeployments } from "@/app/(dashboard)/hooks/models/useModels";
import ComplexityRouterConfig, {
  ComplexityRouterConfigValue,
  DEFAULT_ADAPTIVE_WEIGHTS,
  DEFAULT_SESSION_AFFINITY,
  DEFAULT_DEPLOYMENT_AFFINITY,
  DEFAULT_TIER_DISTANCE_PENALTY,
} from "./ComplexityRouterConfig";
import { KeywordTierRule } from "./KeywordTierRules";
import { DEFAULT_ESCALATION_KEYWORDS } from "./EscalationKeywords";
import { DEFAULT_MATCH_THRESHOLD } from "./SemanticKeywordMatching";
import {
  BuildComplexityRouterConfigParams,
  buildComplexityRouterConfig,
  getKeywordTierRulesError,
  getClassifierModelError,
  getMissingTiersError,
  getPlanModeTierError,
  getSemanticConfigError,
  getTierLabelsError,
} from "./build_complexity_router_config";
import { activeTierName, activeTierRows, resolveComplexityDefaultModel } from "./tier_rows";
import { DEFAULT_TIER_LABELS } from "./complexity_router_tiers";
import type { ComplexityTier } from "./KeywordTierRules";
import { buildAutoRouterTestTargets, AutoRouterTestTarget } from "./build_auto_router_test_targets";
import AutoRouterConnectionTest from "./auto_router_connection_test";
import AutoRouterRoutingTest from "./AutoRouterRoutingTest";
import { toast } from "@/lib/toast";
import {
  getAllPresets,
  getPresetByKey,
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
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

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

// getAllPresets() already returns a stable, module-level array (see autorouter_presets.ts), so
// this is resolved once at import time rather than re-called from inside the component every render.
const presets = getAllPresets();

// A one-line summary of what's configured, shown when the detailed section is collapsed so a
// caller can see the shape of the config without opening it.
const tierConfigSummary = (config: ComplexityRouterConfigValue): string => {
  const parts = activeTierRows(config)
    .filter((row) => row.models.length > 0)
    .map((row) => `${DEFAULT_TIER_LABELS[row.id as ComplexityTier] ?? activeTierName(row)}: ${row.models.join(", ")}`);
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
  availability: ModelAvailability,
): string | null =>
  getMissingTiersError(activeTierRows(config)) ??
  getTierLabelsError(config.tier_labels) ??
  getPlanModeTierError(config.plan_mode_min_tier, activeTierRows(config)) ??
  getKeywordTierRulesError(keywordTierRules, activeTierRows(config)) ??
  getClassifierModelError(config) ??
  getReferencedModelsError(referencedModelsParams, availability);

const autoRouterSchema = (requiresTeamScope: boolean) =>
  z.object({
    auto_router_name: z.string().min(1, "Auto router name is required"),
    team_id: requiresTeamScope ? z.string().min(1, "Please select a team to continue") : z.string(),
    model_access_group: z.array(z.string()).optional(),
  });

type AddAutoRouterFormValues = z.infer<ReturnType<typeof autoRouterSchema>>;

const EMPTY_FORM_VALUES: AddAutoRouterFormValues = {
  auto_router_name: "",
  team_id: "",
  model_access_group: undefined,
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

const teamScopePayload = (requiresTeamScope: boolean, teamId: string): { team_id?: string } =>
  requiresTeamScope ? { team_id: teamId } : {};

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
}) => {
  const requiresTeamScope = createScope === "team-required";
  const form = useZodForm(autoRouterSchema(requiresTeamScope), { defaultValues: EMPTY_FORM_VALUES });
  const watchedName = useWatch({ control: form.control, name: "auto_router_name" });
  const watchedTeamId = useWatch({ control: form.control, name: "team_id" });
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
  const [showValidationErrors, setShowValidationErrors] = useState<boolean>(false);

  const [selectedPreset, setSelectedPreset] = useState<string | undefined>(undefined);
  // Closed by default: a caller opens it deliberately, either by clicking it or by choosing Custom
  // (which expands it automatically, since there's nothing else to show them their config from). A
  // preset re-collapses it after prefilling, offering the same "here's what got filled in, expand to
  // change it" affordance. A caller can always toggle it manually at any point.
  const [detailsExpanded, setDetailsExpanded] = useState<boolean>(false);

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
    queryKey: ["availableModels", "autoRouter", accessToken],
    queryFn: () => fetchAvailableModels(accessToken),
    enabled: Boolean(accessToken),
  });
  const { data: deployments, isLoading: deploymentsLoading } = useQuery({
    queryKey: autoRouterListKey(userId ?? "", userRole),
    queryFn: () => fetchAllModelDeployments(accessToken, userId ?? "", userRole),
    enabled: Boolean(accessToken),
  });
  const modelsLoading = groupsLoading || deploymentsLoading;
  const modelInfo = React.useMemo(() => data ?? [], [data]);
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
    [presetAvailability],
  );

  const templateItems = React.useMemo(
    () => [
      ...sortedPresetOptions.map(({ preset }) => ({ value: preset.key, label: preset.label })),
      { value: "custom", label: "Custom Configuration" },
    ],
    [sortedPresetOptions],
  );

  const applyPrefill = (prefill: PresetPrefill) => {
    setComplexityRouterConfig(prefill.complexityRouterConfig);
    setCustomTechnicalKeywords(prefill.customTechnicalKeywords);
    setKeywordTierRules(prefill.keywordTierRules);
    setSemanticMatchingEnabled(prefill.semanticMatchingEnabled);
    setEmbeddingModel(prefill.embeddingModel);
    setMatchThreshold(prefill.matchThreshold);
    setEscalationKeywords(prefill.escalationKeywords);
  };

  const handlePresetChange = (presetKey: string | undefined) => {
    if (!presetKey || presetKey === "custom") {
      setSelectedPreset(presetKey);
      applyPrefill(buildEmptyPrefill());
      setDetailsExpanded(true);
      return;
    }

    const preset = getPresetByKey(presetKey);
    // Refuse to apply a preset whose models are not verified available. The dropdown disables
    // these options, so this is a guard against a stale click resolving after the list changed.
    if (!preset) return;
    const presetState = presetAvailability(preset);
    if (presetState.kind !== "available") return;

    setSelectedPreset(presetKey);
    applyPrefill(buildPresetPrefill(preset.complexity_router_config, availability));
    setDetailsExpanded(presetState.viaDeployments);
  };

  const referencedModelsParams = {
    tiers: complexityRouterConfig.tiers,
    classifierType: complexityRouterConfig.classifier_type,
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
  );

  const complexityRouterConfigParams: BuildComplexityRouterConfigParams = {
    tiers: complexityRouterConfig.tiers,
    defaultModel: complexityRouterConfig.default_model,
    planModeMinTier: complexityRouterConfig.plan_mode_min_tier,
    heuristicFirstMaxTier: complexityRouterConfig.heuristic_first_max_tier,
    tierLabels: complexityRouterConfig.tier_labels,
    classifierType: complexityRouterConfig.classifier_type,
    classifierLlmConfig: complexityRouterConfig.classifier_llm_config,
    classifierContextWindowSize: complexityRouterConfig.classifier_context_window_size,
    classifierContextBudgetChars: complexityRouterConfig.classifier_context_budget_chars,
    classifierContextIncludeAssistantTurns: complexityRouterConfig.classifier_context_include_assistant_turns,
    classifierFallback: complexityRouterConfig.classifier_fallback,
    sessionAffinity: complexityRouterConfig.session_affinity ?? DEFAULT_SESSION_AFFINITY,
    deploymentAffinity: complexityRouterConfig.deployment_affinity ?? DEFAULT_DEPLOYMENT_AFFINITY,
    customTechnicalKeywords,
    keywordTierRules,
    semanticMatchingEnabled,
    embeddingModel,
    matchThreshold,
    escalationKeywords,
    adaptive: complexityRouterConfig.adaptive ?? false,
    adaptiveWeights: complexityRouterConfig.adaptive_weights ?? DEFAULT_ADAPTIVE_WEIGHTS,
    tierDistancePenalty: complexityRouterConfig.tier_distance_penalty ?? DEFAULT_TIER_DISTANCE_PENALTY,
    adaptiveEligible: complexityRouterConfig.adaptive_eligible ?? "all",
    returnRawModelName: complexityRouterConfig.return_raw_model_name ?? false,
    tierModelParams: complexityRouterConfig.tier_model_params,
    tierBoundaries: complexityRouterConfig.tier_boundaries,
    tokenThresholds: complexityRouterConfig.token_thresholds,
    dimensionWeights: complexityRouterConfig.dimension_weights,
    reasoningOverrideMinScore: complexityRouterConfig.reasoning_override_min_score,
  };

  const submitRecommendedRouter = async (name: string) => {
    const { tiers } = complexityRouterConfigParams;

    // The one answer the submit button reads, so a disabled button and a refused submit cannot
    // disagree about why. The handler needs it in its own right: the form fires this on Enter
    // regardless of the button's disabled state.
    const blockedReason =
      getSubmitBlockedReason(
        complexityRouterConfig,
        keywordTierRules,
        referencedModelsParams,
        groupsOnlyAvailability,
      ) ?? getSemanticConfigError({ semanticMatchingEnabled, embeddingModel, keywordTierRules });
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
    const submitValues: AddAutoRouterValues = {
      auto_router_name: name,
      ...teamScopePayload(requiresTeamScope, form.getValues("team_id")),
      auto_router_default_model: defaultModel,
      model_type: "complexity_router",
      complexity_router_config: buildComplexityRouterConfig(complexityRouterConfigParams),
      model_access_group: form.getValues("model_access_group"),
    };

    handleAddAutoRouterSubmit(submitValues, accessToken, () => form.reset(EMPTY_FORM_VALUES), handleOk);
  };

  const handleAutoRouterSubmit = async () => {
    const name = form.getValues("auto_router_name");
    if (!name) {
      setShowValidationErrors(true);
      void form.trigger("auto_router_name");
      toast.fromError("Please enter an Auto Router Name");
      return;
    }

    await submitRecommendedRouter(name);
  };

  const handleTestConnection = () => {
    const testTargetParams = {
      tiers: activeTierRows(complexityRouterConfig).map(
        (row) => [activeTierName(row), row.models] as [string, string[]],
      ),
      semanticMatchingEnabled,
      embeddingModel,
      defaultModel: resolveComplexityDefaultModel(complexityRouterConfig, complexityRouterConfig.default_model),
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

  return (
    <TooltipProvider>
      <Card>
        <CardContent>
          <form onSubmit={form.handleSubmit(() => handleAutoRouterSubmit())} noValidate>
            <FieldGroup>
              <FormField
                control={form.control}
                name="auto_router_name"
                label={labelWithHint("Auto Router Name", "Unique name for this auto router configuration")}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="e.g., smart_router, auto_router_1" />}
              </FormField>

              <div>
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
                            {disabledHint && <div className={`text-xs mt-1 ${hintClass}`}>{disabledHint}</div>}
                            {matchedHint && <div className="text-xs mt-1 text-success">{matchedHint}</div>}
                          </div>
                        </SelectItem>
                      );
                    })}
                    <SelectItem value="custom" label="Custom Configuration">
                      <div>
                        <div className="font-medium">Custom Configuration</div>
                        <div className="text-xs text-muted-foreground">Define your auto router from scratch</div>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
                {modelsUnverifiable && (
                  <div className="text-xs mt-1 text-destructive">
                    Could not load available models.{" "}
                    <button type="button" className="underline" onClick={() => refetchModels()}>
                      Retry
                    </button>
                  </div>
                )}
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
                  {({ id, value, onChange }) => <TeamDropdown id={id} value={value} onChange={onChange} />}
                </FormField>
              )}

              <div className="border border-border rounded-lg">
                <button
                  type="button"
                  onClick={() => setDetailsExpanded((expanded) => !expanded)}
                  className="w-full flex flex-col gap-1 px-4 py-3 text-left hover:bg-muted"
                  data-testid="detailed-configuration-toggle"
                >
                  <span className="flex items-center gap-2 font-medium text-foreground">
                    {detailsExpanded ? (
                      <ChevronDown className="size-3 text-muted-foreground" />
                    ) : (
                      <ChevronRight className="size-3 text-muted-foreground" />
                    )}
                    Detailed Configuration
                  </span>
                  {!detailsExpanded && (
                    <span className="text-xs text-muted-foreground line-clamp-2">
                      {tierConfigSummary(complexityRouterConfig)}
                    </span>
                  )}
                </button>
                {detailsExpanded && (
                  <div className="px-4 pb-4">
                    <ComplexityRouterConfig
                      modelInfo={modelInfo}
                      value={complexityRouterConfig}
                      onChange={setComplexityRouterConfig}
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
                      showValidationErrors={showValidationErrors}
                    />
                  </div>
                )}
              </div>

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

              <div className="flex justify-between items-center">
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
                <div className="flex gap-2">
                  <BlockedReasonTooltip reason={submitBlockedReason}>
                    <Button
                      type="button"
                      variant="outline"
                      data-testid="auto-router-test-routing-btn"
                      disabled={submitBlockedReason !== null}
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
                  <BlockedReasonTooltip reason={submitBlockedReason}>
                    <Button
                      type="button"
                      disabled={submitBlockedReason !== null}
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
              defaultModel={resolveComplexityDefaultModel(complexityRouterConfig, complexityRouterConfig.default_model)}
              routerName={watchedName}
              teamId={requiresTeamScope ? watchedTeamId : undefined}
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

      <Dialog
        open={isTestModalVisible}
        onOpenChange={(open) => {
          if (!open) {
            setIsTestModalVisible(false);
            setIsTestingConnection(false);
          }
        }}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle>Connection Test Results</DialogTitle>
          </DialogHeader>
          {isTestModalVisible && (
            <AutoRouterConnectionTest
              key={connectionTestId}
              accessToken={accessToken}
              targets={testTargets}
              onTestComplete={() => setIsTestingConnection(false)}
            />
          )}
          <DialogFooter>
            {" "}
            <Button
              variant="outline"
              onClick={() => {
                setIsTestModalVisible(false);
                setIsTestingConnection(false);
              }}
            >
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  );
};

export default AddAutoRouterTab;
