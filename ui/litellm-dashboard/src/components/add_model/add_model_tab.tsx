import React from "react";
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect } from "antd";
import type { FormInstance } from "antd";
import type { UploadProps } from "antd/es/upload";
import LiteLLMModelNameField from "./litellm_model_name";
import ConditionalPublicModelName from "./conditional_public_model_name";
import ProviderSpecificFields from "./provider_specific_fields";
import AdvancedSettings from "./advanced_settings";
import { Providers, providerLogoMap, getPlaceholder } from "../provider_info_helpers";
import type { Team } from "../key_team_helpers/key_list";

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
}) => {
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

                  <ProviderSpecificFields
                    selectedProvider={selectedProvider}
                    uploadProps={uploadProps}
                  />
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
                    <Button htmlType="submit">Add Model</Button>
                  </div>
                </>
              </Form>
            </Card>
      
      
    </>
  );
};

export default AddModelTab; 