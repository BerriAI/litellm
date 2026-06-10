import React, { useEffect, useState } from "react";
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect, Modal, Radio, Badge, Space } from "antd";
import type { FormInstance } from "antd";
import { Text, TextInput } from "@tremor/react";
import { useTranslation } from "react-i18next";
import { modelAvailableCall } from "../networking";
import ConnectionErrorDisplay from "./model_connection_test";
import { all_admin_roles } from "@/utils/roles";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";
import RouterConfigBuilder from "./RouterConfigBuilder";
import ComplexityRouterConfig from "./ComplexityRouterConfig";
import NotificationManager from "../molecules/notifications_manager";
import { ThunderboltOutlined, BranchesOutlined } from "@ant-design/icons";

interface AddAutoRouterTabProps {
  form: FormInstance;
  handleOk: () => void;
  accessToken: string;
  userRole: string;
}

type RouterType = "complexity" | "semantic";

interface ComplexityTiers {
  SIMPLE: string;
  MEDIUM: string;
  COMPLEX: string;
  REASONING: string;
}

const { Title, Link } = Typography;

const AddAutoRouterTab: React.FC<AddAutoRouterTabProps> = ({ form, handleOk, accessToken, userRole }) => {
  const { t } = useTranslation();
  // State for connection testing
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [showCustomDefaultModel, setShowCustomDefaultModel] = useState<boolean>(false);
  const [showCustomEmbeddingModel, setShowCustomEmbeddingModel] = useState<boolean>(false);

  // Router type state - default to complexity router
  const [routerType, setRouterType] = useState<RouterType>("complexity");

  // Semantic router config (existing)
  const [routerConfig, setRouterConfig] = useState<any>(null);

  // Complexity router config (new)
  const [complexityTiers, setComplexityTiers] = useState<ComplexityTiers>({
    SIMPLE: "",
    MEDIUM: "",
    COMPLEX: "",
    REASONING: "",
  });

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
        console.log("Fetched models for auto router:", uniqueModels);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info for auto router:", error);
      }
    };
    loadModels();
  }, [accessToken]);

  const isAdmin = all_admin_roles.includes(userRole);

  // Test connection when button is clicked
  const handleTestConnection = async () => {
    setIsTestingConnection(true);
    setConnectionTestId(`test-${Date.now()}`);
    setIsResultModalVisible(true);
  };

  // Auto router specific form submit handler
  const handleAutoRouterSubmit = () => {
    console.log("Auto router submit triggered!");
    console.log("Router type:", routerType);

    const currentFormValues = form.getFieldsValue();
    console.log("Form values:", currentFormValues);

    // Check basic required fields first
    if (!currentFormValues.auto_router_name) {
      NotificationManager.fromBackend(t("addModel.addAutoRouterTab.enterRouterNameError"));
      return;
    }

    // Validation differs based on router type
    if (routerType === "complexity") {
      // Complexity Router validation
      const filledTiers = Object.values(complexityTiers).filter(Boolean);
      if (filledTiers.length === 0) {
        NotificationManager.fromBackend(t("addModel.addAutoRouterTab.selectTierModelError"));
        return;
      }

      // For complexity router, use the first non-empty tier as default
      const defaultModel =
        complexityTiers.MEDIUM || complexityTiers.SIMPLE || complexityTiers.COMPLEX || complexityTiers.REASONING;

      // Set form values for complexity router
      form.setFieldsValue({
        custom_llm_provider: "auto_router",
        model: currentFormValues.auto_router_name,
        api_key: "not_required_for_auto_router",
        auto_router_default_model: defaultModel,
      });

      form
        .validateFields(["auto_router_name"])
        .then((values) => {
          console.log("Complexity router validation passed");

          // Build the complexity router config
          const submitValues = {
            ...values,
            auto_router_name: currentFormValues.auto_router_name,
            auto_router_default_model: defaultModel,
            // Use special model prefix for complexity router
            model_type: "complexity_router",
            complexity_router_config: {
              tiers: complexityTiers,
            },
            model_access_group: currentFormValues.model_access_group,
          };

          console.log("Final submit values:", submitValues);
          handleAddAutoRouterSubmit(submitValues, accessToken, form, handleOk);
        })
        .catch((error) => {
          console.error("Validation failed:", error);
          NotificationManager.fromBackend(t("addModel.addAutoRouterTab.fillRequiredFieldsError"));
        });
    } else {
      // Semantic Router validation (existing logic)
      if (!currentFormValues.auto_router_default_model) {
        NotificationManager.fromBackend(t("addModel.addAutoRouterTab.selectDefaultModelError"));
        return;
      }

      form.setFieldsValue({
        custom_llm_provider: "auto_router",
        model: currentFormValues.auto_router_name,
        api_key: "not_required_for_auto_router",
      });

      // Custom validation for router config
      if (!routerConfig || !routerConfig.routes || routerConfig.routes.length === 0) {
        NotificationManager.fromBackend(t("addModel.addAutoRouterTab.configureRouteError"));
        return;
      }

      // Check if all routes have required fields
      const invalidRoutes = routerConfig.routes.filter(
        (route: any) => !route.name || !route.description || route.utterances.length === 0,
      );

      if (invalidRoutes.length > 0) {
        NotificationManager.fromBackend(t("addModel.addAutoRouterTab.invalidRoutesError"));
        return;
      }

      form
        .validateFields()
        .then((values) => {
          console.log("Form validation passed, submitting with values:", values);
          const submitValues = {
            ...values,
            auto_router_config: routerConfig,
            model_type: "semantic_router",
          };
          console.log("Final submit values:", submitValues);
          handleAddAutoRouterSubmit(submitValues, accessToken, form, handleOk);
        })
        .catch((error) => {
          console.error("Validation failed:", error);
          const fieldErrors = error.errorFields || [];
          if (fieldErrors.length > 0) {
            const missingFields = fieldErrors.map((field: any) => {
              const fieldName = field.name[0];
              const friendlyNames: { [key: string]: string } = {
                auto_router_name: t("addModel.addAutoRouterTab.autoRouterNameLabel"),
                auto_router_default_model: t("addModel.addAutoRouterTab.defaultModelLabel"),
                auto_router_embedding_model: t("addModel.addAutoRouterTab.embeddingModelLabel"),
              };
              return friendlyNames[fieldName] || fieldName;
            });
            NotificationManager.fromBackend(
              t("addModel.addAutoRouterTab.missingFieldsError", { fields: missingFields.join(", ") }),
            );
          } else {
            NotificationManager.fromBackend(t("addModel.addAutoRouterTab.fillRequiredFieldsError"));
          }
        });
    }
  };

  return (
    <>
      <Title level={2}>{t("addModel.addAutoRouterTab.title")}</Title>
      <Text className="text-gray-600 mb-6">{t("addModel.addAutoRouterTab.subtitle")}</Text>

      <Card className="mb-4">
        <div className="mb-4">
          <Text className="text-sm font-medium mb-2 block">{t("addModel.addAutoRouterTab.routerTypeLabel")}</Text>
          <Radio.Group value={routerType} onChange={(e) => setRouterType(e.target.value)} className="w-full">
            <Space direction="vertical" className="w-full">
              <Radio value="complexity" className="w-full">
                <div className="flex items-center gap-2">
                  <ThunderboltOutlined className="text-yellow-500" />
                  <span className="font-medium">{t("addModel.addAutoRouterTab.complexityRouterLabel")}</span>
                  <Badge
                    count={t("addModel.addAutoRouterTab.complexityRouterBadge")}
                    style={{
                      backgroundColor: "#52c41a",
                      fontSize: "10px",
                      padding: "0 6px",
                    }}
                  />
                </div>
                <div className="text-xs text-gray-500 ml-6 mt-1">
                  {t("addModel.addAutoRouterTab.complexityRouterDesc")}
                  <br />
                  <span className="text-green-600">
                    ✓ {t("addModel.addAutoRouterTab.complexityRouterZeroApi")}
                  </span> ·{" "}
                  <span className="text-green-600">✓ {t("addModel.addAutoRouterTab.complexityRouterLatency")}</span> ·{" "}
                  <span className="text-green-600">✓ {t("addModel.addAutoRouterTab.complexityRouterNoCost")}</span>
                </div>
              </Radio>
              <Radio value="semantic" className="w-full mt-2">
                <div className="flex items-center gap-2">
                  <BranchesOutlined className="text-blue-500" />
                  <span className="font-medium">{t("addModel.addAutoRouterTab.semanticRouterLabel")}</span>
                </div>
                <div className="text-xs text-gray-500 ml-6 mt-1">
                  {t("addModel.addAutoRouterTab.semanticRouterDesc")}
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
          {/* Auto Router Name */}
          <Form.Item
            rules={[{ required: true, message: t("addModel.addAutoRouterTab.autoRouterNameRequired") }]}
            label={t("addModel.addAutoRouterTab.autoRouterNameLabel")}
            name="auto_router_name"
            tooltip={t("addModel.addAutoRouterTab.autoRouterNameTooltip")}
            labelCol={{ span: 10 }}
            labelAlign="left"
          >
            <TextInput placeholder={t("addModel.addAutoRouterTab.autoRouterNamePlaceholder")} />
          </Form.Item>

          {/* Conditional rendering based on router type */}
          {routerType === "complexity" ? (
            /* Complexity Router Configuration */
            <div className="w-full mb-4">
              <ComplexityRouterConfig
                modelInfo={modelInfo}
                value={complexityTiers}
                onChange={(tiers) => {
                  setComplexityTiers(tiers);
                }}
              />
            </div>
          ) : (
            /* Semantic Router Configuration (existing) */
            <>
              {/* Router Configuration Builder */}
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

              {/* Auto Router Default Model */}
              <Form.Item
                rules={[
                  { required: routerType === "semantic", message: t("addModel.addAutoRouterTab.defaultModelRequired") },
                ]}
                label={t("addModel.addAutoRouterTab.defaultModelLabel")}
                name="auto_router_default_model"
                tooltip={t("addModel.addAutoRouterTab.defaultModelTooltip")}
                labelCol={{ span: 10 }}
                labelAlign="left"
              >
                <AntdSelect
                  placeholder={t("addModel.addAutoRouterTab.defaultModelPlaceholder")}
                  onChange={(value) => {
                    setShowCustomDefaultModel(value === "custom");
                  }}
                  options={[
                    ...Array.from(new Set(modelInfo.map((option) => option.model_group))).map((model_group) => ({
                      value: model_group,
                      label: model_group,
                    })),
                    { value: "custom", label: t("addModel.addAutoRouterTab.enterCustomModelName") },
                  ]}
                  style={{ width: "100%" }}
                  showSearch={true}
                />
              </Form.Item>

              {/* Auto Router Embedding Model */}
              <Form.Item
                label={t("addModel.addAutoRouterTab.embeddingModelLabel")}
                name="auto_router_embedding_model"
                tooltip={t("addModel.addAutoRouterTab.embeddingModelTooltip")}
                labelCol={{ span: 10 }}
                labelAlign="left"
              >
                <AntdSelect
                  value={form.getFieldValue("auto_router_embedding_model")}
                  placeholder={t("addModel.addAutoRouterTab.embeddingModelPlaceholder")}
                  onChange={(value) => {
                    setShowCustomEmbeddingModel(value === "custom");
                    form.setFieldValue("auto_router_embedding_model", value);
                  }}
                  options={[
                    ...Array.from(new Set(modelInfo.map((option) => option.model_group))).map((model_group) => ({
                      value: model_group,
                      label: model_group,
                    })),
                    { value: "custom", label: t("addModel.addAutoRouterTab.enterCustomModelName") },
                  ]}
                  style={{ width: "100%" }}
                  showSearch={true}
                  allowClear
                />
              </Form.Item>
            </>
          )}

          <div className="flex items-center my-4">
            <div className="flex-grow border-t border-gray-200"></div>
            <span className="px-4 text-gray-500 text-sm">
              {t("addModel.addAutoRouterTab.additionalSettingsDivider")}
            </span>
            <div className="flex-grow border-t border-gray-200"></div>
          </div>

          {/* Model Access Groups - Admin only */}
          {isAdmin && (
            <Form.Item
              label={t("addModel.addAutoRouterTab.modelAccessGroupLabel")}
              name="model_access_group"
              className="mb-4"
              tooltip={t("addModel.addAutoRouterTab.modelAccessGroupTooltip")}
            >
              <AntdSelect
                mode="tags"
                showSearch
                placeholder={t("addModel.addAutoRouterTab.modelAccessGroupPlaceholder")}
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
            <Tooltip title={t("addModel.addAutoRouterTab.needHelpTooltip")}>
              <Typography.Link href="https://github.com/BerriAI/litellm/issues">
                {t("addModel.addAutoRouterTab.needHelp")}
              </Typography.Link>
            </Tooltip>
            <div className="space-x-2">
              <Button onClick={handleTestConnection} loading={isTestingConnection}>
                {t("addModel.addAutoRouterTab.testConnectionButton")}
              </Button>
              <Button
                type="primary"
                onClick={() => {
                  console.log("Add Auto Router button clicked!");
                  handleAutoRouterSubmit();
                }}
              >
                {t("addModel.addAutoRouterTab.addAutoRouterButton")}
              </Button>
            </div>
          </div>
        </Form>
      </Card>

      {/* Test Connection Results Modal */}
      <Modal
        title={t("addModel.addAutoRouterTab.connectionTestResultsTitle")}
        open={isResultModalVisible}
        onCancel={() => {
          setIsResultModalVisible(false);
          setIsTestingConnection(false);
        }}
        footer={[
          <Button
            key="close"
            onClick={() => {
              setIsResultModalVisible(false);
              setIsTestingConnection(false);
            }}
          >
            {t("common.close")}
          </Button>,
        ]}
        width={700}
      >
        {/* Only render the ConnectionErrorDisplay when modal is visible and we have a test ID */}
        {isResultModalVisible && (
          <ConnectionErrorDisplay
            key={connectionTestId}
            formValues={form.getFieldsValue()}
            accessToken={accessToken}
            testMode="chat"
            modelName={form.getFieldValue("auto_router_name")}
            onClose={() => {
              setIsResultModalVisible(false);
              setIsTestingConnection(false);
            }}
            onTestComplete={() => setIsTestingConnection(false)}
          />
        )}
      </Modal>
    </>
  );
};

export default AddAutoRouterTab;
