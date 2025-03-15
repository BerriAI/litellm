import React, { useState } from "react";
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect, Modal } from "antd";
import type { FormInstance } from "antd";
import type { UploadProps } from "antd/es/upload";
import LiteLLMModelNameField from "./litellm_model_name";
import ConditionalPublicModelName from "./conditional_public_model_name";
import ProviderSpecificFields from "./provider_specific_fields";
import AdvancedSettings from "./advanced_settings";
import { Providers, providerLogoMap, getPlaceholder } from "../provider_info_helpers";
import type { Team } from "../key_team_helpers/key_list";
import { CredentialItem } from "../networking";
import ConnectionErrorDisplay from "./ConnectionErrorDisplay";
import { TEST_MODES } from "./add_model_modes";

interface AddModelTabProps {
  form: FormInstance;
  handleOk: () => void;
  selectedProvider: Providers;
  setSelectedProvider: (provider: Providers) => void;
  providerModels: string[];
  setProviderModelsFn: (provider: Providers) => void;
  getPlaceholder: (provider: Providers) => string;
  uploadProps: UploadProps;
  showAdvancedSettings: boolean;
  setShowAdvancedSettings: (show: boolean) => void;
  teams: Team[] | null;
  credentials: CredentialItem[];
  accessToken: string;
}

const { Title, Link } = Typography;

const AddModelTab: React.FC<AddModelTabProps> = ({
  form,
  handleOk,
  selectedProvider,
  setSelectedProvider,
  providerModels,
  setProviderModelsFn,
  getPlaceholder,
  uploadProps,
  showAdvancedSettings,
  setShowAdvancedSettings,
  teams,
  credentials,
  accessToken,
}) => {
  // Add state for test mode and connection error
  const [testMode, setTestMode] = useState<string>("chat");
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);

  // Test connection directly when button is clicked
  const handleTestConnection = async () => {
    setIsTestingConnection(true);
    setIsResultModalVisible(true);
    // The actual testing is handled in ConnectionErrorDisplay component
  };

  return (
    <>
      <Title level={2}>Add new model</Title>
      <Card>
        <Form
          form={form}
          onFinish={handleOk}
          labelCol={{ span: 10 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            {/* Provider Selection */}
            <Form.Item
              rules={[{ required: true, message: "Required" }]}
              label="Provider:"
              name="custom_llm_provider"
              tooltip="E.g. OpenAI, Azure OpenAI, Anthropic, Bedrock, etc."
              labelCol={{ span: 10 }}
              labelAlign="left"
            >
              <AntdSelect
                showSearch={true}
                value={selectedProvider}
                onChange={(value) => {
                  setSelectedProvider(value);
                  setProviderModelsFn(value);
                  form.setFieldsValue({ 
                    model: [],
                    model_name: undefined 
                  });
                }}
              >
                {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => (
                  <AntdSelect.Option
                    key={providerEnum}
                    value={providerEnum}
                  >
                    <div className="flex items-center space-x-2">
                      <img
                        src={providerLogoMap[providerDisplayName]}
                        alt={`${providerEnum} logo`}
                        className="w-5 h-5"
                        onError={(e) => {
                          // Create a div with provider initial as fallback
                          const target = e.target as HTMLImageElement;
                          const parent = target.parentElement;
                          if (parent) {
                            const fallbackDiv = document.createElement('div');
                            fallbackDiv.className = 'w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs';
                            fallbackDiv.textContent = providerDisplayName.charAt(0);
                            parent.replaceChild(fallbackDiv, target);
                          }
                        }}
                      />
                      <span>{providerDisplayName}</span>
                    </div>
                  </AntdSelect.Option>
                ))}
              </AntdSelect>
            </Form.Item>
            <LiteLLMModelNameField
                selectedProvider={selectedProvider}
                providerModels={providerModels}
                getPlaceholder={getPlaceholder}
              />
            
            {/* Conditionally Render "Public Model Name" */}
            <ConditionalPublicModelName  />
                        
            {/* Select Mode */}
            <Form.Item
              label="Mode"
              name="mode"
              tooltip="Optional - When mode is set litellm will use this for health checks. If mode is `embedding` then /embeddings will be used when trying to check health."
            >
              <AntdSelect
                style={{ width: '100%' }}
                value={testMode}
                onChange={(value) => setTestMode(value)}
                options={TEST_MODES}
              />
            </Form.Item>

            {/* Credentials */}
            <div className="mb-4">
              <Typography.Text className="text-sm text-gray-500 mb-2">
                Either select existing credentials OR enter new provider credentials below
              </Typography.Text>
            </div>

            <Form.Item
              label="Existing Credentials"
              name="litellm_credential_name"
            >
              <AntdSelect
                showSearch
                placeholder="Select or search for existing credentials"
                optionFilterProp="children"
                filterOption={(input, option) =>
                  (option?.label ?? '').toLowerCase().includes(input.toLowerCase())
                }
                options={[
                  { value: null, label: 'None' },
                  ...credentials.map((credential) => ({
                    value: credential.credential_name,
                    label: credential.credential_name
                  }))
                ]}
                allowClear
              />
            </Form.Item>

            <div className="flex items-center my-4">
              <div className="flex-grow border-t border-gray-200"></div>
              <span className="px-4 text-gray-500 text-sm">OR</span>
              <div className="flex-grow border-t border-gray-200"></div>
            </div>

            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) => 
                prevValues.litellm_credential_name !== currentValues.litellm_credential_name ||
                prevValues.provider !== currentValues.provider
              }
            >
              {({ getFieldValue }) => {
                const credentialName = getFieldValue('litellm_credential_name');
                console.log("🔑 Credential Name Changed:", credentialName);
                // Only show provider specific fields if no credentials selected
                if (!credentialName) {
                  return (
                    <ProviderSpecificFields
                      selectedProvider={selectedProvider}
                      uploadProps={uploadProps}
                    />
                  );
                }
                return (
                  <div className="text-gray-500 text-sm text-center">
                    Using existing credentials - no additional provider fields needed
                  </div>
                );
              }}
            </Form.Item>
            <AdvancedSettings 
              showAdvancedSettings={showAdvancedSettings}
              setShowAdvancedSettings={setShowAdvancedSettings}
              teams={teams}
            />

            <div className="flex justify-between items-center mb-4">
              <Tooltip title="Get help on our github">
                <Typography.Link href="https://github.com/BerriAI/litellm/issues">
                  Need Help?
                </Typography.Link>
              </Tooltip>
              <div className="space-x-2">
                <Button onClick={handleTestConnection} loading={isTestingConnection}>Test Connect</Button>
                <Button htmlType="submit">Add Model</Button>
              </div>
            </div>
          </>
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
        <ConnectionErrorDisplay 
          formValues={form.getFieldsValue()}
          accessToken={accessToken}
          testMode={testMode}
          modelName={form.getFieldValue('model_name') || form.getFieldValue('model')}
          onClose={() => {
            setIsResultModalVisible(false);
            setIsTestingConnection(false);
          }}
        />
      </Modal>
    </>
  );
};

export default AddModelTab; 