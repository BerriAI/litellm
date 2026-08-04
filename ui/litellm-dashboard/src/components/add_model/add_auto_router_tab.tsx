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
import { DEFAULT_ESCALATION_KEYWORDS } from "./EscalationKeywords";
import { DEFAULT_MATCH_THRESHOLD } from "./SemanticKeywordMatching";
import {
  buildComplexityRouterConfig,
  getKeywordTierRulesError,
  getMissingTiersError,
  getSemanticConfigError,
} from "./build_complexity_router_config";
import { buildAutoRouterTestTargets, AutoRouterTestTarget } from "./build_auto_router_test_targets";
import AutoRouterConnectionTest from "./auto_router_connection_test";
import NotificationManager from "../molecules/notifications_manager";

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

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
      setModelAccessGroups(response["data"].map((model: any) => model["id"]));
    };
    fetchModelAccessGroups();
  }, [accessToken]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info for auto router:", error);
      }
    };
    loadModels();
  }, [accessToken]);

  const isAdmin = all_admin_roles.includes(userRole);

  const modelGroupOptions = Array.from(new Set(modelInfo.map((option) => option.model_group))).map((model_group) => ({
    value: model_group,
    label: model_group,
  }));

  // Why the submit is unavailable, or null when it is available. The button reads this to disable
  // itself and to say what is missing, so the two can never give different answers.
  const submitBlockedReason =
    getMissingTiersError(complexityRouterConfig.tiers) ?? getKeywordTierRulesError(keywordTierRules);

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
