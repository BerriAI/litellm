import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { all_admin_roles } from "@/utils/roles";
import { Switch, Tab, TabGroup, TabList, TabPanel, TabPanels, Text } from "@tremor/react";
import type { FormInstance } from "antd";
import { Select as AntdSelect, Button, Card, Col, Form, Modal, Row, Tooltip, Typography } from "antd";
import type { UploadProps } from "antd/es/upload";
import React, { useEffect, useMemo, useState } from "react";
import TeamDropdown from "../common_components/team_dropdown";
import type { Team } from "../key_team_helpers/key_list";
import {
  type CredentialItem,
  type ProviderCreateInfo,
  getGuardrailsList,
  modelAvailableCall,
  tagListCall,
} from "../networking";
import { Providers, providerLogoMap } from "../provider_info_helpers";
import { Tag } from "../tag_management/types";
import AddAutoRouterTab from "./add_auto_router_tab";
import { TEST_MODES } from "./add_model_modes";
import AdvancedSettings from "./advanced_settings";
import ConditionalPublicModelName from "./conditional_public_model_name";
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit";
import LiteLLMModelNameField from "./litellm_model_name";
import ConnectionErrorDisplay from "./model_connection_test";
import ProviderSpecificFields from "./provider_specific_fields";

interface AddModelTabProps {
  form: FormInstance; // For the Add Model tab
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
  const [testMode, setTestMode] = useState<string>("chat");
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [tagsList, setTagsList] = useState<Record<string, Tag>>({});
  // Using a unique ID to force the ConnectionErrorDisplay to remount and run a fresh test
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  // Provider metadata for driving the provider select from backend config
  const {
    data: providerMetadata,
    isLoading: isProviderMetadataLoading,
    error: providerMetadataError,
  } = useProviderFields();

  useEffect(() => {
    const fetchGuardrails = async () => {
      try {
        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    fetchGuardrails();
  }, [accessToken]);

  useEffect(() => {
    const fetchTags = async () => {
      try {
        const response = await tagListCall(accessToken);
        setTagsList(response);
      } catch (error) {
        console.error("Failed to fetch tags:", error);
      }
    };

    fetchTags();
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

  const sortedProviderMetadata: ProviderCreateInfo[] = useMemo(() => {
    if (!providerMetadata) {
      return [];
    }
    return [...providerMetadata].sort((a, b) => a.provider_display_name.localeCompare(b.provider_display_name));
  }, [providerMetadata]);

  const providerMetadataErrorText = providerMetadataError
    ? providerMetadataError instanceof Error
      ? providerMetadataError.message
      : "Failed to load providers"
    : null;

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
                      showSearch
                      loading={isProviderMetadataLoading}
                      placeholder={isProviderMetadataLoading ? "Loading providers..." : "Select a provider"}
                      optionFilterProp="data-label"
                      onChange={(value) => {
                        setSelectedProvider(value as Providers);
                        setProviderModelsFn(value as Providers);
                        form.setFieldsValue({
                          custom_llm_provider: value,
                        });
                        form.setFieldsValue({
                          model: [],
                          model_name: undefined,
                        });
                      }}
                    >
                      {providerMetadataErrorText && sortedProviderMetadata.length === 0 && (
                        <AntdSelect.Option key="__error" value="">
                          {providerMetadataErrorText}
                        </AntdSelect.Option>
                      )}
                      {sortedProviderMetadata.map((providerInfo) => {
                        const displayName = providerInfo.provider_display_name;
                        const providerKey = providerInfo.provider;
                        const logoSrc = providerLogoMap[displayName] ?? "";

                        return (
                          <AntdSelect.Option key={providerKey} value={providerKey} data-label={displayName}>
                            <div className="flex items-center space-x-2">
                              {logoSrc ? (
                                <img
                                  src={logoSrc}
                                  alt={`${displayName} logo`}
                                  className="w-5 h-5"
                                  onError={(e) => {
                                    const target = e.currentTarget as HTMLImageElement;
                                    const parent = target.parentElement;
                                    if (!parent || !parent.contains(target)) {
                                      return;
                                    }

                                    try {
                                      const fallbackDiv = document.createElement("div");
                                      fallbackDiv.className =
                                        "w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs";
                                      fallbackDiv.textContent = displayName.charAt(0);
                                      parent.replaceChild(fallbackDiv, target);
                                    } catch (error) {
                                      console.error("Failed to replace provider logo fallback:", error);
                                    }
                                  }}
                                />
                              ) : (
                                <div className="w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs">
                                  {displayName.charAt(0)}
                                </div>
                              )}
                              <span>{displayName}</span>
                            </div>
                          </AntdSelect.Option>
                        );
                      })}
                    </AntdSelect>
                  </Form.Item>
                  <LiteLLMModelNameField
                    selectedProvider={selectedProvider}
                    providerModels={providerModels}
                    getPlaceholder={getPlaceholder}
                  />

                  {/* Conditionally Render "Public Model Name" */}
                  <ConditionalPublicModelName />

                  {/* Select Mode */}
                  <Form.Item label="Mode" name="mode" className="mb-1">
                    <AntdSelect
                      style={{ width: "100%" }}
                      value={testMode}
                      onChange={(value) => setTestMode(value)}
                      options={TEST_MODES}
                    />
                  </Form.Item>
                  <Row>
                    <Col span={10}></Col>
                    <Col span={10}>
                      <Text className="mb-5 mt-1">
                        <strong>Optional</strong> - LiteLLM endpoint to use when health checking this model{" "}
                        <Link href="https://docs.litellm.ai/docs/proxy/health#health" target="_blank">
                          Learn more
                        </Link>
                      </Text>
                    </Col>
                  </Row>

                  {/* Credentials */}
                  <div className="mb-4">
                    <Typography.Text className="text-sm text-gray-500 mb-2">
                      Either select existing credentials OR enter new provider credentials below
                    </Typography.Text>
                  </div>

                  <Form.Item label="Existing Credentials" name="litellm_credential_name" initialValue={null}>
                    <AntdSelect
                      showSearch
                      placeholder="Select or search for existing credentials"
                      optionFilterProp="children"
                      filterOption={(input, option) =>
                        (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                      }
                      options={[
                        { value: null, label: "None" },
                        ...credentials.map((credential) => ({
                          value: credential.credential_name,
                          label: credential.credential_name,
                        })),
                      ]}
                      allowClear
                    />
                  </Form.Item>

                  <Form.Item
                    noStyle
                    shouldUpdate={(prevValues, currentValues) =>
                      prevValues.litellm_credential_name !== currentValues.litellm_credential_name ||
                      prevValues.provider !== currentValues.provider
                    }
                  >
                    {({ getFieldValue }) => {
                      const credentialName = getFieldValue("litellm_credential_name");
                      console.log("🔑 Credential Name Changed:", credentialName);
                      // Only show provider specific fields if no credentials selected
                      if (!credentialName) {
                        return (
                          <>
                            <div className="flex items-center my-4">
                              <div className="flex-grow border-t border-gray-200"></div>
                              <span className="px-4 text-gray-500 text-sm">OR</span>
                              <div className="flex-grow border-t border-gray-200"></div>
                            </div>
                            <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />
                          </>
                        );
                      }
                      return null;
                    }}
                  </Form.Item>
                  <div className="flex items-center my-4">
                    <div className="flex-grow border-t border-gray-200"></div>
                    <span className="px-4 text-gray-500 text-sm">Additional Model Info Settings</span>
                    <div className="flex-grow border-t border-gray-200"></div>
                  </div>
                  {/* Team-only Model Switch */}
                  <Form.Item
                    label="Team-BYOK Model"
                    tooltip="Only use this model + credential combination for this team. Useful when teams want to onboard their own OpenAI keys."
                    className="mb-4"
                  >
                    <Tooltip
                      title={
                        !premiumUser
                          ? "This is an enterprise-only feature. Upgrade to premium to restrict model+credential combinations to a specific team."
                          : ""
                      }
                      placement="top"
                    >
                      <Switch
                        checked={isTeamOnly}
                        onChange={(checked) => {
                          setIsTeamOnly(checked);
                          if (!checked) {
                            form.setFieldValue("team_id", undefined);
                          }
                        }}
                        disabled={!premiumUser}
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
                          message: "Please select a team.",
                        },
                      ]}
                    >
                      <TeamDropdown teams={teams} disabled={!premiumUser} />
                    </Form.Item>
                  )}
                  {isAdmin && (
                    <>
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
                          tokenSeparators={[","]}
                          options={modelAccessGroups.map((group) => ({
                            value: group,
                            label: group,
                          }))}
                          maxTagCount="responsive"
                          allowClear
                        />
                      </Form.Item>
                    </>
                  )}
                  <AdvancedSettings
                    showAdvancedSettings={showAdvancedSettings}
                    setShowAdvancedSettings={setShowAdvancedSettings}
                    teams={teams}
                    guardrailsList={guardrailsList}
                    tagsList={tagsList}
                  />

                  <div className="flex justify-between items-center mb-4">
                    <Tooltip title="Get help on our github">
                      <Typography.Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Typography.Link>
                    </Tooltip>
                    <div className="space-x-2">
                      <Button onClick={handleTestConnection} loading={isTestingConnection}>
                        Test Connect
                      </Button>
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
            // The key prop tells React to create a fresh component instance when it changes
            key={connectionTestId}
            formValues={form.getFieldsValue()}
            accessToken={accessToken}
            testMode={testMode}
            modelName={form.getFieldValue("model_name") || form.getFieldValue("model")}
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
