import React, { useEffect, useMemo, useState } from "react";
import { Card, Form, Button, Tooltip, Typography, Select as AntdSelect, Modal, message, Alert, Switch } from "antd";
import type { FormInstance } from "antd";
import type { UploadProps } from "antd/es/upload";
import { TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import LiteLLMModelNameField from "./litellm_model_name";
import ConditionalPublicModelName from "./conditional_public_model_name";
import ProviderSpecificFields from "./provider_specific_fields";
import AdvancedSettings from "./advanced_settings";
import { Providers, providerLogoMap, getPlaceholder } from "../provider_info_helpers";
import type { Team } from "../key_team_helpers/key_list";
import { CredentialItem, getGuardrailsList, modelAvailableCall } from "../networking";
import ConnectionErrorDisplay from "./model_connection_test";
import { TEST_MODES } from "./add_model_modes";
import { Row, Col } from "antd";

import TeamDropdown from "../common_components/team_dropdown";
import { all_admin_roles } from "@/utils/roles";
import AddAutoRouterTab from "./add_auto_router_tab";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import { InfoCircleOutlined } from "@ant-design/icons";

interface AddModelTabProps {
  form: FormInstance; // For the Add Model tab
  handleOk: () => void;
  selectedProvider: Providers | null;
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
  userRole: string;
  premiumUser: boolean;
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
  userRole,
  premiumUser,
}) => {
  // Create separate form instance for auto router
  const [autoRouterForm] = Form.useForm();
  // State for test mode and connection testing
  const [testMode, setTestMode] = useState<string | null>(null);
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  // Using a unique ID to force the ConnectionErrorDisplay to remount and run a fresh test
  const [connectionTestId, setConnectionTestId] = useState<string>("");




  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map(
          (g: { guardrail_name: string }) => g.guardrail_name
        );
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    fetchGuardrails();
  }, [accessToken]);
  
  // Test connection when button is clicked
  const handleTestConnection = async () => {
    setIsTestingConnection(true);
    // Generate a new test ID (using timestamp for uniqueness)
    // This forces React to create a new instance of ConnectionErrorDisplay
    setConnectionTestId(`test-${Date.now()}`);
    // Show the modal with the fresh test
    setIsResultModalVisible(true);
  };

  // State for team-only switch
  const [isTeamOnly, setIsTeamOnly] = useState<boolean>(false);

  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
      setModelAccessGroups(response["data"].map((model: any) => model["id"]));
    };
    fetchModelAccessGroups();
  }, [accessToken]);

  const isAdmin = all_admin_roles.includes(userRole);

  const handleAutoRouterOk = () => {
    autoRouterForm
      .validateFields()
      .then((values) => {
        handleAddAutoRouterSubmit(values, accessToken, autoRouterForm, handleOk);
      })
      .catch((error) => {
        console.error("Validation failed:", error);
      });
  };

  return (
    <>
      <TabGroup className="w-full">
        <TabList className="mb-4">
          <Tab>Add Model</Tab>
          <Tab>Add Auto Router</Tab>
        </TabList>
        <TabPanels>
          <TabPanel>
            <Title level={2}>Add Model</Title>
            

            
            <Card>
        <Form
          form={form}
          onFinish={(values) => {
            console.log("🔥 Form onFinish triggered with values:", values);
            handleOk();
          }}
          onFinishFailed={(errorInfo) => {
            console.log("💥 Form onFinishFailed triggered:", errorInfo);
          }}
          labelCol={{ span: 10 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            {/* Model Configuration Section - Grouped together */}
            <div className="border border-gray-300 rounded-lg p-6 mb-6 ">

              {/* Model Mappings Title Section */}
              <div className="mb-6">
                <div className="flex items-center gap-2">
                  <h4 className="text-lg font-semibold text-gray-900 mb-0">Model Mappings</h4>
                  <Tooltip title="Map public model names to LiteLLM model names for load balancing">
                    <InfoCircleOutlined className="text-gray-400" />
                  </Tooltip>
                </div>
                <p className="text-sm text-gray-600 mt-1">
                  Configure how your models will be mapped for{" "}
                  <Link
                    href="https://docs.litellm.ai/docs/proxy/load_balancing"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800"
                  >
                    load balancing
                  </Link>
                </p>
              </div>

              {/* Provider and Model Selection - Side by side */}
              <Row gutter={16} className="mb-6">
                <Col span={12}>
                  <Form.Item
                    rules={[{ required: true, message: "Required" }]}
                    label="Provider:"
                    name="custom_llm_provider"
                    tooltip="E.g. OpenAI, Azure OpenAI, Anthropic, Bedrock, etc."
                    labelCol={{ span: 24 }}
                    labelAlign="left"
                  >
                    <AntdSelect
                      showSearch={true}
                      value={selectedProvider}
                      placeholder="Select a provider"
                      getPopupContainer={(triggerNode) => triggerNode.parentElement}
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
                </Col>
                <Col span={12}>
                  <LiteLLMModelNameField
                    selectedProvider={selectedProvider}
                    providerModels={providerModels}
                    getPlaceholder={getPlaceholder}
                  />
                </Col>
              </Row>
              
              {/* Model Mappings Table - Below selections */}
              <div>
                <ConditionalPublicModelName providerModels={providerModels} showTitle={false} />
              </div>
            </div>

            {/* Authentication Section - Side-by-Side Approach */}
            <div className="border border-gray-300 rounded-lg p-6 mb-6 ">
              <div className="mb-4">
                <h4 className="text-lg font-semibold text-gray-900 mb-1">Authentication & Connection</h4>
                <p className="text-sm text-gray-600">Configure authentication and connection settings for the model</p>
              </div>
              
              {/* Mode Selection */}
              <Form.Item
                label="Mode"
                name="mode"
                className="mb-4"
                labelCol={{ span: 24 }}
                wrapperCol={{ span: 24 }}
                tooltip={
                  <>
                    <strong>Optional</strong> - LiteLLM endpoint to use when health checking this model{" "}
                    <a href="https://docs.litellm.ai/docs/proxy/health#health" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:text-blue-300">
                      Learn more
                    </a>
                  </>
                }
              >
                <AntdSelect
                  style={{ width: '50%' }}
                  value={testMode}
                  onChange={(value) => setTestMode(value)}
                  allowClear
                  placeholder="Select a mode (optional)"
                  options={[
                    { value: null, label: 'None (Default)' },
                    ...TEST_MODES
                  ]}
                />
              </Form.Item>
              
              {/* Side-by-Side Credentials Section */}
              <div className="flex gap-6">
                {/* Left Side - New Credentials */}
                <div className="flex-1 border border-gray-200 rounded-lg p-4">
                  <div className="mb-2">
                    <h5 className="text-md font-semibold text-gray-900 mb-1">New Credentials</h5>
                  </div>
                  
                  <Form.Item noStyle shouldUpdate>
                    {({ getFieldValue }) => {
                      const credentialName = getFieldValue('litellm_credential_name');
                      const hasProvider = selectedProvider;
                      
                      if (credentialName) {
                        return (
                          <div className="bg-gray-50 border border-gray-200 rounded p-3 text-sm text-gray-600 text-center">
                            Using existing credentials - new credentials not needed
                          </div>
                        );
                      }
                      
                      if (!hasProvider) {
                        return (
                          <div className="bg-blue-50 border border-blue-200 rounded p-3 text-sm text-blue-700 text-center">
                            Select a provider above to configure new credentials
                          </div>
                        );
                      }
                      
                      return (
                        <div className="space-y-3">
                          <ProviderSpecificFields
                            selectedProvider={selectedProvider}
                            uploadProps={uploadProps}
                            labelCol={{ span: 24 }}
                            wrapperCol={{ span: 24 }}
                          />
                        </div>
                      );
                    }}
                  </Form.Item>
                </div>
                
                {/* Right Side - Existing Credentials */}
                <div className={`flex-1 border border-gray-200 rounded-lg p-4 ${credentials.length === 0 ? 'bg-gray-50' : ''}`}>
                  <div className="mb-2">
                    <h5 className={`text-md font-semibold mb-1 ${credentials.length === 0 ? 'text-gray-500' : 'text-gray-900'}`}>
                      Existing Credentials
                    </h5>
                  </div>
                  
                  <div className="space-y-3">
                    <Form.Item
                      label="Choose from Saved"
                      name="litellm_credential_name"
                      labelCol={{ span: 24 }}
                      wrapperCol={{ span: 24 }}
                      className={credentials.length === 0 ? 'opacity-60' : ''}
                      tooltip={credentials.length === 0 
                        ? 'No existing credentials available' 
                        : 'Select previously saved credentials, or leave empty to enter new ones'
                      }
                    >
                      <AntdSelect
                        showSearch
                        style={{ width: '50%' }}
                        placeholder={credentials.length === 0 
                          ? 'No credentials available' 
                          : 'Select or search for existing credentials'
                        }
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
                        disabled={credentials.length === 0}
                        getPopupContainer={(triggerNode) => triggerNode.parentElement}
                      />
                    </Form.Item>
                    
                    <Form.Item noStyle shouldUpdate>
                      {({ getFieldValue }) => {
                        const credentialName = getFieldValue('litellm_credential_name');
                        
                        if (credentialName) {
                          return (
                            <div className="bg-green-50 border border-green-200 rounded p-3 text-sm text-green-700">
                              ✓ Using existing credentials: <strong>{credentialName}</strong>
                            </div>
                          );
                        }
                        
                        return null;
                      }}
                    </Form.Item>
                  </div>
                </div>
              </div>
            </div>
            {/* Additional Model Info Settings Section */}
            <div className="border border-gray-300 rounded-lg p-6 mb-6 ">
              <div className="mb-4">
                <h4 className="text-lg font-semibold text-gray-900 mb-1">Additional Model Info Settings</h4>
                <p className="text-sm text-gray-600">Configure team access and model grouping options</p>
              </div>

              {/* Team-BYOK Model Switch */}
              <Form.Item
                label="Team-BYOK Model"
                tooltip={
                  <div>
                    <div className="mb-2">
                      <strong>Team-BYOK (Bring Your Own Key) Model:</strong> Makes this model + credential combination exclusive to the selected team only.
                    </div>
                    <div className="mb-2">
                      <strong>Key Differences:</strong>
                    </div>
                    <div className="mb-1">
                      • <strong>Regular models:</strong> Can be shared across multiple teams (if you give them access)
                    </div>
                    <div className="mb-2">
                      • <strong>Team-BYOK models:</strong> Exclusive to one team - only keys belonging to that team can call this model
                    </div>
                    <div className="mb-2">
                      <strong>Use Case:</strong> Perfect when teams want to add their own API keys (e.g., their own OpenAI keys) to the proxy without sharing them with other teams.
                    </div>
                    <div>
                      <strong>Note:</strong> You can still add regular models to teams in LiteLLM OSS - this just provides additional exclusivity.{" "}
                      <Link
                        href="https://docs.litellm.ai/docs/proxy/team_alias"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 hover:text-blue-300"
                      >
                        Learn more
                      </Link>
                    </div>
                  </div>
                }
                className="mb-4"
              >
                <Tooltip 
                  title={!premiumUser ? "This is an enterprise-only feature. Upgrade to premium to restrict model+credential combinations to a specific team." : ""}
                  placement="top"
                >
                  <Switch 
                    checked={isTeamOnly}
                    onChange={(checked: boolean) => {
                      setIsTeamOnly(checked);
                      if (!checked) {
                        form.setFieldValue('team_id', undefined);
                      }
                    }}
                    disabled={!premiumUser}
                    style={{
                      backgroundColor: isTeamOnly ? '#1890ff' : '#d9d9d9',
                      borderColor: '#1890ff',
                      boxShadow: isTeamOnly ? '0 0 0 2px rgba(24, 144, 255, 0.2)' : 'none'
                    }}
                    className="border-2"
                  />
                </Tooltip>
              </Form.Item>

              {/* Conditional Team Selection */}
              {isTeamOnly && (
                <Form.Item
                  label="Select Team"
                  name="team_id"
                  className="mb-4"
                  tooltip="Only keys for this team will be able to call this model."
                  rules={[
                    {
                      required: isTeamOnly && !isAdmin,
                      message: 'Please select a team.'
                    }
                  ]}
                >
                  <TeamDropdown teams={teams} disabled={!premiumUser} />
                </Form.Item>
              )}

              {/* Model Access Group (Admin only) */}
              {isAdmin && (
                <Form.Item
                  label="Model Access Group"
                  name="model_access_group"
                  className="mb-4"
                  tooltip="Use model access groups to give users access to select models, and add new ones to the group over time."
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
            </div>
            <AdvancedSettings 
              showAdvancedSettings={showAdvancedSettings}
              setShowAdvancedSettings={setShowAdvancedSettings}
              teams={teams}
              guardrailsList={guardrailsList}
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
          </TabPanel>
          <TabPanel>
            <AddAutoRouterTab
              form={autoRouterForm}
              handleOk={handleAutoRouterOk}
              accessToken={accessToken}
              userRole={userRole}
            />
          </TabPanel>
        </TabPanels>
      </TabGroup>
      
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
            // The key prop tells React to create a fresh component instance when it changes
            key={connectionTestId}
            formValues={form.getFieldsValue()}
            accessToken={accessToken}
            testMode={testMode || "chat"}
            modelName={form.getFieldValue('model_name') || form.getFieldValue('model')}
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

export default AddModelTab; 