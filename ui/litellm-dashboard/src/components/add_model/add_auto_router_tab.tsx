import React, { useEffect, useState } from "react";
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect, Modal } from "antd";
import type { FormInstance } from "antd";
import { Text, TextInput } from "@tremor/react";
import { modelAvailableCall } from "../networking";
import ConnectionErrorDisplay from "./model_connection_test";
import { all_admin_roles } from "@/utils/roles";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";
import RouterConfigBuilder from "./RouterConfigBuilder";
import NotificationManager from "../molecules/notifications_manager";

interface AddAutoRouterTabProps {
  form: FormInstance;
  handleOk: () => void;
  accessToken: string;
  userRole: string;
}

const { Title, Link } = Typography;

const AddAutoRouterTab: React.FC<AddAutoRouterTabProps> = ({ form, handleOk, accessToken, userRole }) => {
  // State for connection testing
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [showCustomDefaultModel, setShowCustomDefaultModel] = useState<boolean>(false);
  const [showCustomEmbeddingModel, setShowCustomEmbeddingModel] = useState<boolean>(false);
  const [routerConfig, setRouterConfig] = useState<any>(null);

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
    console.log("Router config:", routerConfig);
    const currentFormValues = form.getFieldsValue();
    console.log("Form values:", currentFormValues);

    // Check basic required fields first
    if (!currentFormValues.auto_router_name) {
      NotificationManager.fromBackend("Please enter an Auto Router Name");
      return;
    }

    if (!currentFormValues.auto_router_default_model) {
      NotificationManager.fromBackend("Please select a Default Model");
      return;
    }

    // Set auto router specific form values that are required by the regular model form
    form.setFieldsValue({
      custom_llm_provider: "auto_router",
      model: currentFormValues.auto_router_name,
      // api_key is not needed for auto router, but form expects it
      api_key: "not_required_for_auto_router",
    });

    // Custom validation for router config
    if (!routerConfig || !routerConfig.routes || routerConfig.routes.length === 0) {
      NotificationManager.fromBackend("Please configure at least one route for the auto router");
      return;
    }

    // Check if all routes have required fields
    const invalidRoutes = routerConfig.routes.filter(
      (route: any) => !route.name || !route.description || route.utterances.length === 0,
    );

    if (invalidRoutes.length > 0) {
      NotificationManager.fromBackend(
        "Please ensure all routes have a target model, description, and at least one utterance",
      );
      return;
    }

    form
      .validateFields()
      .then((values) => {
        console.log("Form validation passed, submitting with values:", values);
        // Add the router config to form values
        const submitValues = {
          ...values,
          auto_router_config: routerConfig,
        };
        console.log("Final submit values:", submitValues);
        handleAddAutoRouterSubmit(submitValues, accessToken, form, handleOk);
      })
      .catch((error) => {
        console.error("Validation failed:", error);

        // Extract specific field errors
        const fieldErrors = error.errorFields || [];
        if (fieldErrors.length > 0) {
          const missingFields = fieldErrors.map((field: any) => {
            const fieldName = field.name[0];
            const friendlyNames: { [key: string]: string } = {
              auto_router_name: "Auto Router Name",
              auto_router_default_model: "Default Model",
              auto_router_embedding_model: "Embedding Model",
            };
            return friendlyNames[fieldName] || fieldName;
          });
          NotificationManager.fromBackend(`Please fill in the following required fields: ${missingFields.join(", ")}`);
        } else {
          NotificationManager.fromBackend("Please fill in all required fields");
        }
      });
  };

  return (
    <>
      <Title level={2}>Add Auto Router</Title>
      <Text className="text-gray-600 mb-6">
        Create an auto router with intelligent routing logic that automatically selects the best model based on user
        input patterns and semantic matching.
      </Text>

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
            rules={[{ required: true, message: "Auto router name is required" }]}
            label="Auto Router Name"
            name="auto_router_name"
            tooltip="Unique name for this auto router configuration"
            labelCol={{ span: 10 }}
            labelAlign="left"
          >
            <TextInput placeholder="e.g., auto_router_1, smart_routing" />
          </Form.Item>

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
            rules={[{ required: true, message: "Default model is required" }]}
            label="Default Model"
            name="auto_router_default_model"
            tooltip="Fallback model to use when auto routing logic cannot determine the best model"
            labelCol={{ span: 10 }}
            labelAlign="left"
          >
            <AntdSelect
              placeholder="Select a default model"
              onChange={(value) => {
                setShowCustomDefaultModel(value === "custom");
              }}
              options={[
                ...Array.from(new Set(modelInfo.map((option) => option.model_group))).map((model_group) => ({
                  value: model_group,
                  label: model_group,
                })),
                { value: "custom", label: "Enter custom model name" },
              ]}
              style={{ width: "100%" }}
              showSearch={true}
            />
          </Form.Item>

          {/* Auto Router Embedding Model */}
          <Form.Item
            label="Embedding Model"
            name="auto_router_embedding_model"
            tooltip="Optional: Embedding model to use for semantic routing decisions"
            labelCol={{ span: 10 }}
            labelAlign="left"
          >
            <AntdSelect
              value={form.getFieldValue("auto_router_embedding_model")}
              placeholder="Select an embedding model (optional)"
              onChange={(value) => {
                setShowCustomEmbeddingModel(value === "custom");
                form.setFieldValue("auto_router_embedding_model", value);
              }}
              options={[
                ...Array.from(new Set(modelInfo.map((option) => option.model_group))).map((model_group) => ({
                  value: model_group,
                  label: model_group,
                })),
                { value: "custom", label: "Enter custom model name" },
              ]}
              style={{ width: "100%" }}
              showSearch={true}
              allowClear
            />
          </Form.Item>
          <div className="flex items-center my-4">
            <div className="flex-grow border-t border-gray-200"></div>
            <span className="px-4 text-gray-500 text-sm">Additional Settings</span>
            <div className="flex-grow border-t border-gray-200"></div>
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
              <Button onClick={handleTestConnection} loading={isTestingConnection}>
                Test Connect
              </Button>
              <Button
                onClick={() => {
                  console.log("Add Auto Router button clicked!");
                  console.log("Current router config:", routerConfig);
                  console.log("Current form values:", form.getFieldsValue());
                  handleAutoRouterSubmit();
                }}
              >
                Add Auto Router
              </Button>
            </div>
          </div>
        </Form>
      </Card>

      {/* Test Connection Results Modal */}
      <Modal
        title="Connection Test Results"
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
            Close
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
