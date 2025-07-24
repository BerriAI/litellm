import React, { useEffect, useState } from "react";
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect, Modal, Upload } from "antd";
import type { FormInstance } from "antd";
import type { UploadProps } from "antd/es/upload";
import { UploadOutlined } from "@ant-design/icons";
import { Text, TextInput } from "@tremor/react";
import { Row, Col } from "antd";
import { CredentialItem, modelAvailableCall } from "../networking";
import ConnectionErrorDisplay from "./model_connection_test";
import { all_admin_roles } from "@/utils/roles";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { fetchAvailableModels, ModelGroup } from "../chat_ui/llm_calls/fetch_models";

interface AddAutoRouterTabProps {
  form: FormInstance;
  handleOk: () => void;
  accessToken: string;
  userRole: string;
}

const { Title, Link } = Typography;

const AddAutoRouterTab: React.FC<AddAutoRouterTabProps> = ({
  form,
  handleOk,
  accessToken,
  userRole,
}) => {
  // State for connection testing
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  const [connectionTestId, setConnectionTestId] = useState<string>("");



  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [showCustomDefaultModel, setShowCustomDefaultModel] = useState<boolean>(false);
  const [showCustomEmbeddingModel, setShowCustomEmbeddingModel] = useState<boolean>(false);

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
    form
      .validateFields()
      .then((values) => {
        handleAddAutoRouterSubmit(values, accessToken, form, handleOk);
      })
      .catch((error) => {
        console.error("Validation failed:", error);
      });
  };

  return (
    <>
      <Title level={2}>Add Auto Router</Title>
      <Text className="text-gray-600 mb-6">
        Add an auto router, this allows you to add auto routing logic to automatically select the best model
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

          {/* Auto Router Config Path */}
          <Form.Item
            rules={[{ required: true, message: "Config path is required" }]}
            label="Config Path"
            name="auto_router_config_path"
            tooltip="Path to the router configuration file that defines routing logic"
            labelCol={{ span: 10 }}
            labelAlign="left"
          >
            <TextInput placeholder="e.g., /path/to/router_config.json" />
          </Form.Item>

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
              value={form.getFieldValue('auto_router_default_model')}
              placeholder="Select a default model"
              onChange={(value) => {
                setShowCustomDefaultModel(value === 'custom');
                form.setFieldValue('auto_router_default_model', value);
              }}
              options={[
                ...Array.from(new Set(modelInfo.map(option => option.model_group)))
                  .map((model_group) => ({
                    value: model_group,
                    label: model_group,
                  })),
                { value: 'custom', label: 'Enter custom model name' }
              ]}
              style={{ width: "100%" }}
              showSearch={true}
            />
          </Form.Item>
          {showCustomDefaultModel && (
            <Form.Item
              label="Custom Default Model"
              name="custom_default_model"
              labelCol={{ span: 10 }}
              labelAlign="left"
            >
              <TextInput 
                placeholder="Enter custom model name"
                onChange={(e) => {
                  form.setFieldValue('auto_router_default_model', e.target.value);
                }}
              />
            </Form.Item>
          )}

          {/* Auto Router Embedding Model */}
          <Form.Item
            label="Embedding Model"
            name="auto_router_embedding_model"
            tooltip="Optional: Embedding model to use for semantic routing decisions"
            labelCol={{ span: 10 }}
            labelAlign="left"
          >
            <AntdSelect
              value={form.getFieldValue('auto_router_embedding_model')}
              placeholder="Select an embedding model (optional)"
              onChange={(value) => {
                setShowCustomEmbeddingModel(value === 'custom');
                form.setFieldValue('auto_router_embedding_model', value);
              }}
              options={[
                ...Array.from(new Set(modelInfo.map(option => option.model_group)))
                  .map((model_group) => ({
                    value: model_group,
                    label: model_group,
                  })),
                { value: 'custom', label: 'Enter custom model name' }
              ]}
              style={{ width: "100%" }}
              showSearch={true}
              allowClear
            />
          </Form.Item>
          {showCustomEmbeddingModel && (
            <Form.Item
              label="Custom Embedding Model"
              name="custom_embedding_model"
              labelCol={{ span: 10 }}
              labelAlign="left"
            >
              <TextInput 
                placeholder="Enter custom embedding model name"
                onChange={(e) => {
                  form.setFieldValue('auto_router_embedding_model', e.target.value);
                }}
              />
            </Form.Item>
          )}

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
                tokenSeparators={[',']}
                options={modelAccessGroups.map((group) => ({
                  value: group,
                  label: group
                }))}
                maxTagCount="responsive"
                allowClear
              />
            </Form.Item>
          )}

          <div className="flex justify-between items-center mb-4">
            <Tooltip title="Get help on our github">
              <Typography.Link href="https://github.com/BerriAI/litellm/issues">
                Need Help?
              </Typography.Link>
            </Tooltip>
            <div className="space-x-2">
              <Button onClick={handleTestConnection} loading={isTestingConnection}>Test Connect</Button>
              <Button htmlType="submit">Add Auto Router</Button>
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
          <Button key="close" onClick={() => {
            setIsResultModalVisible(false);
            setIsTestingConnection(false);
          }}>
            Close
          </Button>
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
            modelName={form.getFieldValue('auto_router_name')}
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