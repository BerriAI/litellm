import React, { useEffect, useState } from "react";
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect, Modal } from "antd";
import { TextInput } from "@tremor/react";
import { modelAvailableCall } from "../networking";
import { all_admin_roles } from "@/utils/roles";
import { type ModelWriteScope } from "@/utils/modelPermissions";
import TeamDropdown from "../common_components/team_dropdown";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import ComplexityRouterConfig, {
  ComplexityRouterConfigValue,
  DEFAULT_ADAPTIVE_WEIGHTS,
  DEFAULT_SESSION_AFFINITY,
  DEFAULT_TIER_DISTANCE_PENALTY,
} from "./ComplexityRouterConfig";
import { KeywordTierRule } from "./KeywordTierRules";
import { hydrateKeywordTierRules } from "./complexity_router_keywords";
import { DEFAULT_ESCALATION_KEYWORDS } from "./EscalationKeywords";
import { DEFAULT_MATCH_THRESHOLD } from "./SemanticKeywordMatching";
import {
  buildComplexityRouterConfig,
  getMissingTiersError,
  getSemanticConfigError,
} from "./build_complexity_router_config";
import { buildAutoRouterTestTargets, AutoRouterTestTarget } from "./build_auto_router_test_targets";
import AutoRouterConnectionTest from "./auto_router_connection_test";
import NotificationManager from "../molecules/notifications_manager";
import { getAllPresets, getPresetByKey, getMissingModelsInPreset, AutoRouterPreset } from "@/lib/autorouter_presets";

type PresetAvailability =
  | { kind: "available" }
  | { kind: "loading" }
  | { kind: "unverifiable" }
  | { kind: "missing_models"; models: readonly string[] };

// Every non-"available" state disables the option. Selection derives from this same function
// (see presetAvailability below), so an option a caller can click is always one we can apply.
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

// "loading"/"unverifiable" are transient system states, not a gap specific to this preset;
// only a caller-specific missing-model reason gets the alarming red treatment.
const isPresetHintAlarming = (availability: PresetAvailability): boolean => availability.kind === "missing_models";

interface AddAutoRouterTabProps {
  handleOk: () => void;
  accessToken: string;
  userRole: string;
  /**
   * How this caller must scope what they create. A team admin has to name a team, because
   * POST /model/new rejects an unscoped create from any non-proxy-admin; without the selector
   * their submit is a guaranteed 403.
   */
  createScope?: ModelWriteScope;
}

const { Title } = Typography;

const AddAutoRouterTab: React.FC<AddAutoRouterTabProps> = ({
  handleOk,
  accessToken,
  userRole,
  createScope = "unscoped-ok",
}) => {
  const requiresTeamScope = createScope === "team-required";
  const [form] = Form.useForm();
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [modelsLoadState, setModelsLoadState] = useState<"loading" | "loaded" | "error">("loading");

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

  const [isTestModalVisible, setIsTestModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  const [connectionTestId, setConnectionTestId] = useState<number>(0);
  const [testTargets, setTestTargets] = useState<AutoRouterTestTarget[]>([]);

  const [selectedPreset, setSelectedPreset] = useState<string | undefined>(undefined);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
      setModelAccessGroups(response["data"].map((model: any) => model["id"]));
    };
    fetchModelAccessGroups();
  }, [accessToken]);

  useEffect(() => {
    let ignore = false;

    const loadModels = async () => {
      setModelsLoadState("loading");
      setModelInfo([]);
      setSelectedPreset(undefined);
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        if (ignore) return;
        setModelInfo(uniqueModels);
        setModelsLoadState("loaded");
      } catch (error) {
        console.error("Error fetching model info for auto router:", error);
        if (ignore) return;
        setModelsLoadState("error");
      }
    };
    loadModels();

    return () => {
      ignore = true;
    };
  }, [accessToken]);

  const isAdmin = all_admin_roles.includes(userRole);

  const modelGroupOptions = Array.from(new Set(modelInfo.map((option) => option.model_group))).map((model_group) => ({
    value: model_group,
    label: model_group,
  }));

  const availableModelSet = new Set(modelInfo.map((m) => m.model_group));
  const presets = React.useMemo(() => getAllPresets(), []);

  // A preset's models can only be trusted against a successfully loaded list. Selection and the
  // greyed-out state derive from this one function, so a preset that cannot be selected can never
  // have been applied: while loading we withhold selection rather than let a caller pick a preset
  // whose models we cannot yet verify, and a failed fetch leaves every preset unverifiable. This
  // makes the load-race (pick during loading, then discover a missing model) unrepresentable.
  const presetAvailability = (preset: AutoRouterPreset): PresetAvailability => {
    if (modelsLoadState === "loading") return { kind: "loading" };
    if (modelsLoadState === "error") return { kind: "unverifiable" };
    const missing = getMissingModelsInPreset(preset, availableModelSet);
    return missing.length > 0 ? { kind: "missing_models", models: missing } : { kind: "available" };
  };

  const handlePresetChange = (presetKey: string | undefined) => {
    if (!presetKey || presetKey === "custom") {
      setSelectedPreset(presetKey);
      setComplexityRouterConfig({
        tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
        classifier_type: "heuristic",
      });
      setCustomTechnicalKeywords([]);
      setKeywordTierRules([]);
      setSemanticMatchingEnabled(false);
      setEmbeddingModel(undefined);
      setMatchThreshold(DEFAULT_MATCH_THRESHOLD);
      setEscalationKeywords(DEFAULT_ESCALATION_KEYWORDS);
      return;
    }

    const preset = getPresetByKey(presetKey);
    // Refuse to apply a preset whose models are not verified available. The dropdown disables
    // these options, so this is a guard against a stale click resolving after the list changed.
    if (!preset || presetAvailability(preset).kind !== "available") return;

    setSelectedPreset(presetKey);

    const config = preset.complexity_router_config;
    const presetComplexityRouterConfig: ComplexityRouterConfigValue = {
      tiers: config.tiers,
      classifier_type: config.classifier_type,
      classifier_llm_config: config.classifier_llm_config,
      adaptive: config.adaptive,
      adaptive_weights: config.adaptive_weights,
      tier_distance_penalty: config.tier_distance_penalty,
      adaptive_eligible: config.adaptive_eligible,
      return_raw_model_name: config.return_raw_model_name,
    };
    setComplexityRouterConfig(presetComplexityRouterConfig);

    setCustomTechnicalKeywords(config.custom_technical_keywords ?? []);
    setKeywordTierRules(hydrateKeywordTierRules(config.keyword_tier_rules ?? []));
    setSemanticMatchingEnabled(config.semantic_keyword_matching ?? false);
    setEmbeddingModel(config.embedding_model);
    setMatchThreshold(config.match_threshold ?? DEFAULT_MATCH_THRESHOLD);
    setEscalationKeywords(config.escalation_keywords ?? DEFAULT_ESCALATION_KEYWORDS);
  };

  const submitRecommendedRouter = (name: string) => {
    const {
      tiers,
      classifier_type: classifierType,
      classifier_llm_config: classifierLlmConfig,
      classifier_context_window_size: classifierContextWindowSize,
      classifier_context_per_turn_chars: classifierContextPerTurnChars,
      classifier_context_include_assistant_turns: classifierContextIncludeAssistantTurns,
      session_affinity: sessionAffinity = DEFAULT_SESSION_AFFINITY,
      adaptive = false,
      adaptive_weights: adaptiveWeights = DEFAULT_ADAPTIVE_WEIGHTS,
      tier_distance_penalty: tierDistancePenalty = DEFAULT_TIER_DISTANCE_PENALTY,
      adaptive_eligible: adaptiveEligible = "all",
      return_raw_model_name: returnRawModelName = false,
    } = complexityRouterConfig;

    const missingTiersError = getMissingTiersError(tiers);
    if (missingTiersError) {
      setShowValidationErrors(true);
      NotificationManager.fromBackend(missingTiersError);
      return;
    }

    if (classifierType === "llm" && !classifierLlmConfig?.model) {
      setShowValidationErrors(true);
      NotificationManager.fromBackend("Please select a classifier model, or switch back to Heuristic");
      return;
    }

    const semanticError = getSemanticConfigError({ semanticMatchingEnabled, embeddingModel, keywordTierRules });
    if (semanticError) {
      setShowValidationErrors(true);
      NotificationManager.fromBackend(semanticError);
      return;
    }

    const defaultModel = tiers.MEDIUM[0] || tiers.SIMPLE[0] || tiers.COMPLEX[0] || tiers.REASONING[0];

    form.setFieldsValue({
      custom_llm_provider: "auto_router",
      model: name,
      api_key: "not_required_for_auto_router",
      auto_router_default_model: defaultModel,
    });

    form
      .validateFields(requiresTeamScope ? ["auto_router_name", "team_id"] : ["auto_router_name"])
      .then((values) => {
        const complexityRouterConfigParams = {
          tiers,
          classifierType,
          classifierLlmConfig,
          classifierContextWindowSize,
          classifierContextPerTurnChars,
          classifierContextIncludeAssistantTurns,
          sessionAffinity,
          customTechnicalKeywords,
          keywordTierRules,
          semanticMatchingEnabled,
          embeddingModel,
          matchThreshold,
          escalationKeywords,
          adaptive,
          adaptiveWeights,
          tierDistancePenalty,
          adaptiveEligible,
          returnRawModelName,
        };

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
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-900 mb-2">
              Template <span className="text-red-500">*</span>
            </label>
            <AntdSelect
              value={selectedPreset}
              onChange={handlePresetChange}
              placeholder="Choose a template or select Custom to define your own"
              className="w-full"
              optionLabelProp="label"
              data-testid="template-selector"
            >
              <AntdSelect.Option value="custom" label="Custom Configuration">
                <div>
                  <div className="font-medium">Custom Configuration</div>
                  <div className="text-xs text-gray-500">Define your auto router from scratch</div>
                </div>
              </AntdSelect.Option>
              {presets.map((preset) => {
                const availability = presetAvailability(preset);
                const disabledHint = presetDisabledHint(availability);
                const isDisabled = disabledHint !== null;
                const hintClass = isPresetHintAlarming(availability) ? "text-red-500" : "text-gray-400";

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
                    </div>
                  </AntdSelect.Option>
                );
              })}
            </AntdSelect>
          </div>

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

          <div className="w-full mb-4">
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

          <div className="flex items-center my-4">
            <div className="grow border-t border-gray-200"></div>
            <span className="px-4 text-gray-500 text-sm">Additional Settings</span>
            <div className="grow border-t border-gray-200"></div>
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
              {
                <Button
                  data-testid="auto-router-test-connect-btn"
                  onClick={handleTestConnection}
                  loading={isTestingConnection}
                >
                  Test Connection
                </Button>
              }
              <Button
                type="primary"
                onClick={() => {
                  handleAutoRouterSubmit();
                }}
              >
                Add Auto Router
              </Button>
            </div>
          </div>
        </Form>
      </Card>

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
