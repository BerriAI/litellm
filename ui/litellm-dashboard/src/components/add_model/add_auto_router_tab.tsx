import React, { useEffect, useState } from "react";
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect, Radio, Badge, Space, Modal } from "antd";
import type { FormInstance } from "antd";
import { ThunderboltOutlined, BranchesOutlined } from "@ant-design/icons";
import { Text, TextInput } from "@tremor/react";
import { modelAvailableCall } from "../networking";
import { all_admin_roles } from "@/utils/roles";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import RouterConfigBuilder from "./RouterConfigBuilder";
import ComplexityRouterConfig, {
  ComplexityRouterConfigValue,
  DEFAULT_ADAPTIVE_WEIGHTS,
  DEFAULT_TIER_DISTANCE_PENALTY,
} from "./ComplexityRouterConfig";
import { KeywordTierRule } from "./KeywordTierRules";
import { DEFAULT_MATCH_THRESHOLD } from "./SemanticKeywordMatching";
import {
  buildComplexityRouterConfig,
  getMissingTiersError,
  getSemanticConfigError,
} from "./build_complexity_router_config";
import { buildAutoRouterTestTargets, AutoRouterTestTarget } from "./build_auto_router_test_targets";
import { getSemanticRouterError } from "./build_semantic_router_validation";
import AutoRouterConnectionTest from "./auto_router_connection_test";
import NotificationManager from "../molecules/notifications_manager";

interface AddAutoRouterTabProps {
  form: FormInstance;
  handleOk: () => void;
  accessToken: string;
  userRole: string;
}

type RouterType = "recommended" | "semantic";

const { Title } = Typography;

const AddAutoRouterTab: React.FC<AddAutoRouterTabProps> = ({ form, handleOk, accessToken, userRole }) => {
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);

  const [routerType, setRouterType] = useState<RouterType>("recommended");

  const [complexityRouterConfig, setComplexityRouterConfig] = useState<ComplexityRouterConfigValue>({
    tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    classifier_type: "heuristic",
  });

  const [customTechnicalKeywords, setCustomTechnicalKeywords] = useState<string[]>([]);
  const [keywordTierRules, setKeywordTierRules] = useState<KeywordTierRule[]>([]);
  const [semanticMatchingEnabled, setSemanticMatchingEnabled] = useState<boolean>(false);
  const [embeddingModel, setEmbeddingModel] = useState<string | undefined>(undefined);
  const [matchThreshold, setMatchThreshold] = useState<number>(DEFAULT_MATCH_THRESHOLD);
  const [showValidationErrors, setShowValidationErrors] = useState<boolean>(false);

  // Semantic router config (existing)
  const [routerConfig, setRouterConfig] = useState<any>(null);

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

  const submitRecommendedRouter = (name: string) => {
    const {
      tiers,
      classifier_type: classifierType,
      classifier_llm_config: classifierLlmConfig,
      adaptive = false,
      adaptive_weights: adaptiveWeights = DEFAULT_ADAPTIVE_WEIGHTS,
      tier_distance_penalty: tierDistancePenalty = DEFAULT_TIER_DISTANCE_PENALTY,
      adaptive_eligible: adaptiveEligible = "all",
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
      .validateFields(["auto_router_name"])
      .then((values) => {
        const complexityRouterConfigParams = {
          tiers,
          classifierType,
          classifierLlmConfig,
          customTechnicalKeywords,
          keywordTierRules,
          semanticMatchingEnabled,
          embeddingModel,
          matchThreshold,
          adaptive,
          adaptiveWeights,
          tierDistancePenalty,
          adaptiveEligible,
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

  const submitSemanticRouter = (name: string) => {
    const validationError = getSemanticRouterError({
      defaultModel: form.getFieldValue("auto_router_default_model"),
      embeddingModel: form.getFieldValue("auto_router_embedding_model"),
      routerConfig,
    });
    if (validationError) {
      NotificationManager.fromBackend(validationError);
      return;
    }

    form.setFieldsValue({
      custom_llm_provider: "auto_router",
      model: name,
      api_key: "not_required_for_auto_router",
    });

    form
      .validateFields()
      .then((values) => {
        const submitValues = {
          ...values,
          auto_router_name: name,
          auto_router_config: routerConfig,
          model_type: "semantic_router",
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

    if (routerType === "recommended") {
      submitRecommendedRouter(name);
    } else {
      submitSemanticRouter(name);
    }
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
      <Title level={2}>Add Auto Router</Title>
      <Text className="text-gray-600 mb-6">
        Create an auto router that automatically selects the best model based on request complexity or semantic
        matching. Use in place of a single default model.
      </Text>

      <Card className="mb-4">
        <div className="mb-4">
          <Text className="text-sm font-medium mb-2 block">Router Type</Text>
          <Radio.Group
            value={routerType}
            onChange={(e) => {
              setRouterType(e.target.value);
              setShowValidationErrors(false);
            }}
            className="w-full"
          >
            <Space direction="vertical" className="w-full">
              <Radio value="recommended" className="w-full">
                <div className="flex items-center gap-2">
                  <ThunderboltOutlined className="text-yellow-500" />
                  <span className="font-medium">Auto-Router v2</span>
                  <Badge
                    count="Recommended"
                    style={{ backgroundColor: "#52c41a", fontSize: "10px", padding: "0 6px" }}
                  />
                </div>
                <div className="text-xs text-gray-500 ml-6 mt-1">
                  Routes by request complexity across four tiers, with optional keyword-to-tier overrides and semantic
                  keyword matching. No training data needed.
                </div>
              </Radio>
              <Radio value="semantic" className="w-full mt-2">
                <div className="flex items-center gap-2">
                  <BranchesOutlined className="text-blue-500" />
                  <span className="font-medium">Semantic Router [to be deprecated]</span>
                </div>
                <div className="text-xs text-gray-500 ml-6 mt-1">
                  Routes based on semantic similarity to example utterances. Requires an embedding model and example
                  utterances.
                </div>
              </Radio>
            </Space>
          </Radio.Group>
        </div>
      </Card>

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

          {routerType === "recommended" ? (
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
                showValidationErrors={showValidationErrors}
              />
            </div>
          ) : (
            <>
              <div className="w-full mb-4">
                <RouterConfigBuilder
                  modelInfo={modelInfo}
                  value={routerConfig}
                  onChange={(config) => {
                    setRouterConfig(config);
                    form.setFieldValue("auto_router_config", config);
                  }}
                />
              </div>

              <Form.Item
                rules={[{ required: true, message: "Default model is required" }]}
                label="Default Model"
                name="auto_router_default_model"
                tooltip="Fallback model to use when auto routing logic cannot determine the best model"
                labelCol={{ span: 10 }}
                labelAlign="left"
              >
                <AntdSelect
                  placeholder="Select a default model"
                  options={modelGroupOptions}
                  style={{ width: "100%" }}
                  showSearch
                />
              </Form.Item>

              <Form.Item
                rules={[{ required: true, message: "Embedding model is required" }]}
                label="Embedding Model"
                name="auto_router_embedding_model"
                tooltip="Embedding model to use for semantic routing decisions"
                labelCol={{ span: 10 }}
                labelAlign="left"
              >
                <AntdSelect
                  placeholder="Select an embedding model"
                  options={modelGroupOptions}
                  style={{ width: "100%" }}
                  showSearch
                />
              </Form.Item>
            </>
          )}

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
              {routerType === "recommended" && (
                <Button
                  data-testid="auto-router-test-connect-btn"
                  onClick={handleTestConnection}
                  loading={isTestingConnection}
                >
                  Test Connection
                </Button>
              )}
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
