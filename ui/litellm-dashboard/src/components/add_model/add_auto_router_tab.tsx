import React, { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect, Modal } from "antd";
import { DownOutlined, RightOutlined } from "@ant-design/icons";
import { TextInput } from "@tremor/react";
import { modelAvailableCall } from "../networking";
import { all_admin_roles } from "@/utils/roles";
import { type ModelWriteScope } from "@/utils/modelPermissions";
import TeamDropdown from "../common_components/team_dropdown";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import { autoRouterListKey, fetchAllModelDeployments } from "@/app/(dashboard)/hooks/models/useModels";
import ComplexityRouterConfig, {
  ComplexityRouterConfigValue,
  ComplexityTiers,
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
  getMissingTiersError,
  getSemanticConfigError,
  getTierLabelsError,
} from "./build_complexity_router_config";
import { buildAutoRouterTestTargets, AutoRouterTestTarget } from "./build_auto_router_test_targets";
import AutoRouterConnectionTest from "./auto_router_connection_test";
import AutoRouterRoutingTest from "./AutoRouterRoutingTest";
import NotificationManager from "../molecules/notifications_manager";
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

const resolveDefaultModel = (tiers: ComplexityTiers): string | undefined =>
  tiers.MEDIUM[0] || tiers.SIMPLE[0] || tiers.COMPLEX[0] || tiers.REASONING[0];

// A one-line summary of what's configured, shown when the detailed section is collapsed so a
// caller can see the shape of the config without opening it.
const tierConfigSummary = (tiers: ComplexityTiers): string => {
  const parts = (
    [
      ["Simple", tiers.SIMPLE],
      ["Medium", tiers.MEDIUM],
      ["Complex", tiers.COMPLEX],
      ["Reasoning", tiers.REASONING],
    ] as const
  )
    .filter(([, models]) => models.length > 0)
    .map(([label, models]) => `${label}: ${models.join(", ")}`);
  return parts.length > 0 ? parts.join(" · ") : "No tiers configured yet";
};

// Why the submit is unavailable, or null when it is available. The button reads this to disable
// itself and to say what is missing, so the two can never give different answers. Checks the
// config actually being built, not which preset (if any) it came from: a preset only ever
// prefills once (handlePresetChange), and everything after that is edited exactly like Custom.
const getSubmitBlockedReason = (
  config: ComplexityRouterConfigValue,
  keywordTierRules: KeywordTierRule[],
  referencedModelsParams: Parameters<typeof getReferencedModelsError>[0],
  availability: ModelAvailability,
): string | null =>
  getMissingTiersError(config.tiers) ??
  getTierLabelsError(config.tier_labels) ??
  getKeywordTierRulesError(keywordTierRules) ??
  getReferencedModelsError(referencedModelsParams, availability);

const AddAutoRouterTab: React.FC<AddAutoRouterTabProps> = ({
  handleOk,
  accessToken,
  userRole,
  userId,
  createScope = "unscoped-ok",
}) => {
  const requiresTeamScope = createScope === "team-required";
  const [form] = Form.useForm();
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
  };

  const submitBlockedReason = getSubmitBlockedReason(
    complexityRouterConfig,
    keywordTierRules,
    referencedModelsParams,
    groupsOnlyAvailability,
  );

  const complexityRouterConfigParams: BuildComplexityRouterConfigParams = {
    tiers: complexityRouterConfig.tiers,
    tierLabels: complexityRouterConfig.tier_labels,
    classifierType: complexityRouterConfig.classifier_type,
    classifierLlmConfig: complexityRouterConfig.classifier_llm_config,
    classifierContextWindowSize: complexityRouterConfig.classifier_context_window_size,
    classifierContextPerTurnChars: complexityRouterConfig.classifier_context_per_turn_chars,
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
  };

  const submitRecommendedRouter = (name: string) => {
    const { tiers, tierLabels, classifierType, classifierLlmConfig } = complexityRouterConfigParams;

    const missingTiersError = getMissingTiersError(tiers);
    if (missingTiersError) {
      setShowValidationErrors(true);
      NotificationManager.fromBackend(missingTiersError);
      return;
    }

    const tierLabelsError = getTierLabelsError(tierLabels);
    if (tierLabelsError) {
      setShowValidationErrors(true);
      NotificationManager.fromBackend(tierLabelsError);
      return;
    }

    if (classifierType === "llm" && !classifierLlmConfig?.model) {
      setShowValidationErrors(true);
      NotificationManager.fromBackend("Please select a classifier model, or switch back to Heuristic");
      return;
    }

    const keywordRulesError = getKeywordTierRulesError(keywordTierRules);
    if (keywordRulesError) {
      setShowValidationErrors(true);
      NotificationManager.fromBackend(keywordRulesError);
      return;
    }

    const semanticError = getSemanticConfigError({ semanticMatchingEnabled, embeddingModel, keywordTierRules });
    if (semanticError) {
      setShowValidationErrors(true);
      NotificationManager.fromBackend(semanticError);
      return;
    }

    // submitBlockedReason already disables the button for this, but Form's onFinish (wired to this
    // same handler) fires on Enter regardless of the button's disabled state - without this check,
    // Enter in the name field could still create a router referencing a model that disappeared from
    // availableModelSet after the tiers were filled in.
    const referencedModelsError = getReferencedModelsError(referencedModelsParams, groupsOnlyAvailability);
    if (referencedModelsError) {
      setShowValidationErrors(true);
      NotificationManager.fromBackend(referencedModelsError);
      return;
    }

    const defaultModel = resolveDefaultModel(tiers);

    form.setFieldsValue({
      custom_llm_provider: "auto_router",
      model: name,
      api_key: "not_required_for_auto_router",
      auto_router_default_model: defaultModel,
    });

    form
      .validateFields(requiresTeamScope ? ["auto_router_name", "team_id"] : ["auto_router_name"])
      .then((values) => {
        const submitValues = {
          ...values,
          auto_router_name: name,
          auto_router_default_model: defaultModel,
          model_type: "complexity_router",
          complexity_router_config: buildComplexityRouterConfig(complexityRouterConfigParams),
          model_access_group: form.getFieldValue("model_access_group"),
        };

        handleAddAutoRouterSubmit(submitValues, accessToken, form, handleOk);
      })
      .catch((error) => {
        console.error("Validation failed:", error);
        NotificationManager.fromBackend("Please fill in all required fields");
      });
  };

  const handleAutoRouterSubmit = () => {
    const name = form.getFieldValue("auto_router_name");
    if (!name) {
      setShowValidationErrors(true);
      form.validateFields(["auto_router_name"]).catch(() => undefined);
      NotificationManager.fromBackend("Please enter an Auto Router Name");
      return;
    }

    submitRecommendedRouter(name);
  };

  const handleTestConnection = () => {
    const targets = buildAutoRouterTestTargets({
      tiers: complexityRouterConfig.tiers,
      semanticMatchingEnabled,
      embeddingModel,
    });

    if (targets.length === 0) {
      NotificationManager.fromBackend("Please select at least one model for a complexity tier");
      return;
    }

    setTestTargets(targets);
    setConnectionTestId((id) => id + 1);
    setIsTestingConnection(true);
    setIsTestModalVisible(true);
  };

  return (
    <>
      <Card>
        <Form
          form={form}
          onFinish={handleAutoRouterSubmit}
          labelCol={{ span: 10 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <Form.Item
            rules={[{ required: true, message: "Auto router name is required" }]}
            label="Auto Router Name"
            name="auto_router_name"
            tooltip="Unique name for this auto router configuration"
            labelCol={{ span: 10 }}
            labelAlign="left"
          >
            <TextInput placeholder="e.g., smart_router, auto_router_1" />
          </Form.Item>

          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">Template</label>
            <AntdSelect
              value={selectedPreset}
              onChange={handlePresetChange}
              placeholder="Choose a template or select Custom to define your own"
              className="w-full"
              optionLabelProp="label"
              data-testid="template-selector"
            >
              {sortedPresetOptions.map(({ preset, availability: presetState }) => {
                const disabledHint = presetDisabledHint(presetState);
                const isDisabled = disabledHint !== null;
                const hintClass = isPresetHintAlarming(presetState) ? "text-red-500" : "text-gray-400";
                const matchedHint =
                  presetState.kind === "available" && presetState.viaDeployments ? "Matches your deployments" : null;

                return (
                  <AntdSelect.Option
                    key={preset.key}
                    value={preset.key}
                    label={preset.label}
                    disabled={isDisabled}
                    title={disabledHint ?? preset.description}
                  >
                    <div>
                      <div className="font-medium">{preset.label}</div>
                      <div className="text-xs text-gray-500">{preset.description}</div>
                      {disabledHint && <div className={`text-xs mt-1 ${hintClass}`}>{disabledHint}</div>}
                      {matchedHint && <div className="text-xs mt-1 text-green-600">{matchedHint}</div>}
                    </div>
                  </AntdSelect.Option>
                );
              })}
              <AntdSelect.Option value="custom" label="Custom Configuration">
                <div>
                  <div className="font-medium">Custom Configuration</div>
                  <div className="text-xs text-gray-500">Define your auto router from scratch</div>
                </div>
              </AntdSelect.Option>
            </AntdSelect>
            {modelsUnverifiable && (
              <div className="text-xs mt-1 text-red-500">
                Could not load available models.{" "}
                <button type="button" className="underline" onClick={() => refetchModels()}>
                  Retry
                </button>
              </div>
            )}
          </div>

          {requiresTeamScope && (
            <Form.Item
              label="Select Team"
              name="team_id"
              rules={[{ required: true, message: "Please select a team to continue" }]}
              tooltip="Select the team this auto router belongs to. Only keys for this team will be able to call it."
              labelCol={{ span: 10 }}
              labelAlign="left"
            >
              <TeamDropdown />
            </Form.Item>
          )}

          <div className="border border-gray-200 rounded-lg mb-4">
            <button
              type="button"
              onClick={() => setDetailsExpanded((expanded) => !expanded)}
              className="w-full flex flex-col gap-1 px-4 py-3 text-left hover:bg-gray-50"
              data-testid="detailed-configuration-toggle"
            >
              <span className="flex items-center gap-2 font-medium text-gray-900">
                {detailsExpanded ? (
                  <DownOutlined className="text-xs text-gray-500" />
                ) : (
                  <RightOutlined className="text-xs text-gray-500" />
                )}
                Detailed Configuration
              </span>
              {!detailsExpanded && (
                <span className="text-xs text-gray-500 line-clamp-2">
                  {tierConfigSummary(complexityRouterConfig.tiers)}
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

          {/* Model Access Groups - Admin only */}
          {isAdmin && (
            <Form.Item
              label="Model Access Group"
              name="model_access_group"
              className="mb-4"
              tooltip="Use model access groups to control who can access this auto router"
            >
              <AntdSelect
                mode="tags"
                showSearch
                placeholder="Select existing groups or type to create new ones"
                optionFilterProp="children"
                tokenSeparators={[","]}
                options={modelAccessGroups.map((group) => ({
                  value: group,
                  label: group,
                }))}
                maxTagCount="responsive"
                allowClear
              />
            </Form.Item>
          )}

          <div className="flex justify-between items-center mb-4">
            <Tooltip title="Get help on our github">
              <Typography.Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Typography.Link>
            </Tooltip>
            <div className="space-x-2">
              <Tooltip title={submitBlockedReason}>
                <Button
                  data-testid="auto-router-test-routing-btn"
                  disabled={submitBlockedReason !== null}
                  onClick={() => setIsRoutingTestVisible(true)}
                >
                  Test Routing
                </Button>
              </Tooltip>
              {
                <Button
                  data-testid="auto-router-test-connect-btn"
                  onClick={handleTestConnection}
                  loading={isTestingConnection}
                >
                  Test Connection
                </Button>
              }
              <Tooltip title={submitBlockedReason}>
                <Button
                  type="primary"
                  disabled={submitBlockedReason !== null}
                  onClick={() => {
                    handleAutoRouterSubmit();
                  }}
                >
                  Add Auto Router
                </Button>
              </Tooltip>
            </div>
          </div>
        </Form>
      </Card>

      <Modal
        title="Test Routing"
        open={isRoutingTestVisible}
        destroyOnHidden
        onCancel={() => setIsRoutingTestVisible(false)}
        footer={[
          <Button key="close" onClick={() => setIsRoutingTestVisible(false)}>
            Close
          </Button>,
        ]}
        width={760}
      >
        {isRoutingTestVisible && (
          <AutoRouterRoutingTest
            accessToken={accessToken}
            config={buildComplexityRouterConfig(complexityRouterConfigParams)}
            defaultModel={resolveDefaultModel(complexityRouterConfig.tiers)}
            routerName={form.getFieldValue("auto_router_name")}
            teamId={requiresTeamScope ? form.getFieldValue("team_id") : undefined}
          />
        )}
      </Modal>

      <Modal
        title="Connection Test Results"
        open={isTestModalVisible}
        onCancel={() => {
          setIsTestModalVisible(false);
          setIsTestingConnection(false);
        }}
        footer={[
          <Button
            key="close"
            onClick={() => {
              setIsTestModalVisible(false);
              setIsTestingConnection(false);
            }}
          >
            Close
          </Button>,
        ]}
        width={700}
      >
        {isTestModalVisible && (
          <AutoRouterConnectionTest
            key={connectionTestId}
            accessToken={accessToken}
            targets={testTargets}
            onTestComplete={() => setIsTestingConnection(false)}
          />
        )}
      </Modal>
    </>
  );
};

export default AddAutoRouterTab;
