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

// Define the available test modes
const TEST_MODES = [
  { value: "chat", label: "Chat" },
  { value: "completion", label: "Completion" },
  { value: "embedding", label: "Embedding" },
  { value: "audio_speech", label: "Audio Speech" },
  { value: "audio_transcription", label: "Audio Transcription" },
  { value: "image_generation", label: "Image Generation" },
  { value: "rerank", label: "Rerank" }
];

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
  const [isTestModalVisible, setIsTestModalVisible] = useState<boolean>(false);
  const [connectionError, setConnectionError] = useState<Error | string | null>(null);

  // Show test modal with mode selection
  const showTestModal = () => {
    setConnectionError(null);
    setIsTestModalVisible(true);
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
                <Button onClick={showTestModal}>Test Connect</Button>
                <Button htmlType="submit">Add Model</Button>
              </div>
            </div>
          </>
        </Form>
      </Card>
      
      {/* Test Connection Modal */}
      <Modal
        title="Test Model Connection"
        open={isTestModalVisible}
        onCancel={() => setIsTestModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setIsTestModalVisible(false)}>
            Cancel
          </Button>,
          <Button 
            key="test" 
            type="primary" 
            onClick={() => setConnectionError("Test connection logic is now in ConnectionErrorDisplay")}
            loading={false} // You might want to add a loading state
          >
            Test Connection
          </Button>
        ]}
        width={connectionError ? 700 : 520}
      >
        <div className="mb-4">
          <Typography.Text>Select the mode to test this model with:</Typography.Text>
        </div>
        <AntdSelect
          style={{ width: '100%' }}
          value={testMode}
          onChange={(value) => setTestMode(value)}
          options={TEST_MODES}
        />
        <div className="mt-4">
          <Typography.Text type="secondary">
            Different models support different modes. Choose the appropriate mode for your model.
          </Typography.Text>
        </div>
        
        {/* Render the ConnectionErrorDisplay when there's an error */}
        {connectionError && (
          <div className="mt-4">
            <Typography.Title level={5} type="danger">Connection Test Failed</Typography.Title>
            <div className="border border-red-300 rounded-md overflow-hidden">
              <ConnectionErrorDisplay 
                formValues={form.getFieldsValue()}
                accessToken={accessToken}
                testMode={testMode}
                modelName={form.getFieldValue('model_name') || form.getFieldValue('model')}
                onClose={() => setIsTestModalVisible(false)}
              />
            </div>
          </div>
        )}
      </Modal>
    </>
  );
};

export default AddModelTab; 