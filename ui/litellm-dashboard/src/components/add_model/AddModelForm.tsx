import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { useGuardrails } from "@/app/(dashboard)/hooks/guardrails/useGuardrails";
import { useTags } from "@/app/(dashboard)/hooks/tags/useTags";
import { all_admin_roles, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { Switch, Text } from "@tremor/react";
import type { FormInstance } from "antd";
import { Select as AntdSelect, Button, Card, Col, Form, Input, Modal, Row, Spin, Tooltip, Typography, Alert } from "antd";
import type { UploadProps } from "antd/es/upload";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import TeamDropdown from "../common_components/team_dropdown";
import NotificationsManager from "../molecules/notifications_manager";
import type { Team } from "../key_team_helpers/key_list";
import {
  type CredentialItem,
  type ProviderCreateInfo,
  githubCopilotInitiateAuth,
  githubCopilotCheckStatus,
  chatgptInitiateAuth,
  chatgptCheckStatus,
  modelAvailableCall,
} from "../networking";
import { Providers, providerLogoMap } from "../provider_info_helpers";
import { ProviderLogo } from "../molecules/models/ProviderLogo";
import AdvancedSettings from "./advanced_settings";
import ConditionalPublicModelName from "./conditional_public_model_name";
import LiteLLMModelNameField from "./litellm_model_name";
import ConnectionErrorDisplay from "./model_connection_test";
import ProviderSpecificFields from "./provider_specific_fields";
import { TEST_MODES } from "./add_model_modes";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

interface AddModelFormProps {
  form: FormInstance; // For the Add Model tab
  handleOk: () => Promise<void>;
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
  refetchCredentials?: () => void;
}

const { Title, Link } = Typography;

const AddModelForm: React.FC<AddModelFormProps> = ({
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
  refetchCredentials,
}) => {
  const [testMode, setTestMode] = useState<string>("chat");
  const [isResultModalVisible, setIsResultModalVisible] = useState<boolean>(false);
  const [isTestingConnection, setIsTestingConnection] = useState<boolean>(false);
  // Using a unique ID to force the ConnectionErrorDisplay to remount and run a fresh test
  const [connectionTestId, setConnectionTestId] = useState<string>("");

  const { accessToken, userRole, premiumUser, userId } = useAuthorized();
  const {
    data: providerMetadata,
    isLoading: isProviderMetadataLoading,
    error: providerMetadataError,
  } = useProviderFields();

  // Inline device code flow (GitHub Copilot, ChatGPT, etc.)
  const [ghDeviceCodeState, setGhDeviceCodeState] = useState<
    | { phase: "idle" }
    | { phase: "polling"; deviceCode: string; userCode: string; verificationUri: string }
    | { phase: "success" }
    | { phase: "error"; message: string }
  >({ phase: "idle" });
  // Hold the access_token in a ref — never rendered, injected at submit time
  const ghAccessTokenRef = useRef<string | null>(null);
  const ghPollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const stopGhPolling = useCallback(() => {
    if (ghPollingRef.current) {
      clearTimeout(ghPollingRef.current);
      ghPollingRef.current = null;
    }
  }, []);

  useEffect(() => () => stopGhPolling(), [stopGhPolling]);

  // Determine if the selected provider uses device_code auth flow and get its metadata
  const deviceCodeProviderInfo = React.useMemo(() => {
    if (!providerMetadata) return null;
    const info = providerMetadata.find(
      (p) =>
        p.auth_flow === "device_code" &&
        (p.provider === selectedProvider ||
          p.provider_display_name === Providers[selectedProvider as keyof typeof Providers]),
    );
    return info || null;
  }, [selectedProvider, providerMetadata]);
  const isDeviceCodeProvider = deviceCodeProviderInfo != null;

  // Reset device code state when provider changes away from a device_code provider
  useEffect(() => {
    if (!isDeviceCodeProvider) {
      stopGhPolling();
      setGhDeviceCodeState({ phase: "idle" });
      ghAccessTokenRef.current = null;
    }
  }, [isDeviceCodeProvider, stopGhPolling]);

  const handleGhStartDeviceCode = async () => {
    if (!accessToken || !deviceCodeProviderInfo) return;
    const litellmProvider = deviceCodeProviderInfo.litellm_provider;
    const providerLabel = deviceCodeProviderInfo.provider_display_name || litellmProvider;

    try {
      let deviceId: string;
      let userCode: string;
      let verificationUri: string;
      let pollIntervalMs: number;

      if (litellmProvider === "chatgpt") {
        const result = await chatgptInitiateAuth(accessToken);
        deviceId = result.device_auth_id;
        userCode = result.user_code;
        verificationUri = result.verification_uri;
        pollIntervalMs = result.poll_interval_ms;
      } else {
        const result = await githubCopilotInitiateAuth(accessToken);
        deviceId = result.device_code;
        userCode = result.user_code;
        verificationUri = result.verification_uri;
        pollIntervalMs = result.poll_interval_ms;
      }

      setGhDeviceCodeState({
        phase: "polling",
        deviceCode: deviceId,
        userCode,
        verificationUri,
      });
      let currentPollInterval = pollIntervalMs || 5000;

      const schedulePoll = (delayMs: number) => {
        ghPollingRef.current = setTimeout(async () => {
          try {
            let apiKey: string | undefined;
            let failed = false;
            let errorMsg: string | undefined;
            let retryAfterMs: number | undefined;

            if (litellmProvider === "chatgpt") {
              const status = await chatgptCheckStatus(accessToken, deviceId, userCode);
              console.log(`[${providerLabel} AddModel] poll response:`, status);
              if (status.status === "complete" && status.refresh_token) {
                apiKey = status.refresh_token;
              } else if (status.status === "failed") {
                failed = true;
                errorMsg = status.error;
              }
            } else {
              const status = await githubCopilotCheckStatus(accessToken, deviceId);
              console.log(`[${providerLabel} AddModel] poll response:`, status);
              if (status.status === "complete" && status.access_token) {
                apiKey = status.access_token;
              } else if (status.status === "failed") {
                failed = true;
                errorMsg = status.error;
              } else {
                retryAfterMs = status.retry_after_ms ?? undefined;
              }
            }

            if (apiKey) {
              stopGhPolling();
              ghAccessTokenRef.current = apiKey;
              form.setFieldValue("api_key", apiKey);
              setGhDeviceCodeState({ phase: "success" });
            } else if (failed) {
              stopGhPolling();
              setGhDeviceCodeState({ phase: "error", message: errorMsg || "Authorization failed" });
            } else {
              if (retryAfterMs != null) {
                currentPollInterval = retryAfterMs;
              }
              schedulePoll(currentPollInterval);
            }
          } catch (e) {
            console.error(`[${providerLabel} AddModel] poll error:`, e);
            stopGhPolling();
            setGhDeviceCodeState({ phase: "error", message: "Failed to check authorization status" });
          }
        }, delayMs);
      };
      schedulePoll(currentPollInterval);
    } catch {
      NotificationsManager.error(`Failed to start ${providerLabel} authorization`);
      setGhDeviceCodeState({ phase: "error", message: `Failed to start ${providerLabel} authorization` });
    }
  };

  const handleGhCancel = () => {
    stopGhPolling();
    setGhDeviceCodeState({ phase: "idle" });
    ghAccessTokenRef.current = null;
  };

  const dcProviderLabel = deviceCodeProviderInfo?.provider_display_name || "Provider";

  const renderGhDeviceCodeFlow = () => {
    switch (ghDeviceCodeState.phase) {
      case "idle":
        return (
          <div className="mt-2 text-center">
            <Button
              type="primary"
              onClick={handleGhStartDeviceCode}
            >
              Authorize with {dcProviderLabel}
            </Button>
          </div>
        );
      case "polling":
        return (
          <div className="text-center py-2">
            <Typography.Text className="block mb-2">Enter this code to authorize:</Typography.Text>
            <div
              style={{
                fontSize: "1.8rem",
                fontWeight: "bold",
                fontFamily: "monospace",
                letterSpacing: "0.3em",
                margin: "12px 0",
                padding: "10px 20px",
                background: "#f5f5f5",
                borderRadius: 8,
                display: "inline-block",
                userSelect: "all",
              }}
            >
              {ghDeviceCodeState.userCode}
            </div>
            <div className="mb-3">
              <Button type="link" onClick={() => window.open(ghDeviceCodeState.verificationUri, "_blank")}>
                Open {ghDeviceCodeState.verificationUri}
              </Button>
            </div>
            <Spin />
            <Typography.Text className="block mt-2 mb-3" type="secondary">Waiting for {dcProviderLabel} authorization...</Typography.Text>
            <Button onClick={handleGhCancel}>Cancel</Button>
          </div>
        );
      case "success":
        return (
          <div className="text-center py-2">
            <Typography.Text type="success" className="block mb-2">
              ✓ Authorization complete! Submit the form to add the model.
            </Typography.Text>
          </div>
        );
      case "error":
        return (
          <div className="text-center py-2">
            <Typography.Text type="danger" className="block mb-3">{ghDeviceCodeState.message}</Typography.Text>
            <Button
              style={{ marginRight: 8 }}
              onClick={() => setGhDeviceCodeState({ phase: "idle" })}
            >
              Retry
            </Button>
            <Button onClick={handleGhCancel}>Cancel</Button>
          </div>
        );
    }
  };
  const { data: guardrailsList, isLoading: isGuardrailsLoading, error: guardrailsError } = useGuardrails();
  const { data: tagsList, isLoading: isTagsLoading, error: tagsError } = useTags();

  const handleTestConnection = async () => {
    setIsTestingConnection(true);
    setConnectionTestId(`test-${Date.now()}`);
    setIsResultModalVisible(true);
  };

  const [isTeamOnly, setIsTeamOnly] = useState<boolean>(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  // Team admin specific state
  const [teamAdminSelectedTeam, setTeamAdminSelectedTeam] = useState<string | null>(null);

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
  const isTeamAdmin = isUserTeamAdminForAnyTeam(teams, userId);

  return (
    <>
      <Title level={2}>Add Model</Title>

      <Card>
        <Form
          form={form}
          onFinish={async (values) => {
            console.log("🔥 Form onFinish triggered with values:", values);
            await handleOk().then(() => {
              setTeamAdminSelectedTeam(null);
            });
          }}
          onFinishFailed={(errorInfo) => {
            console.log("💥 Form onFinishFailed triggered:", errorInfo);
          }}
          labelCol={{ span: 10 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            {isTeamAdmin && !isAdmin && (
              <>
                <Form.Item
                  label="Select Team"
                  name="team_id"
                  rules={[{ required: true, message: "Please select a team to continue" }]}
                  tooltip="Select the team for which you want to add this model"
                >
                  <TeamDropdown
                    teams={teams}
                    onChange={(value) => {
                      setTeamAdminSelectedTeam(value);
                    }}
                  />
                </Form.Item>
                {!teamAdminSelectedTeam && (
                  <Alert
                    message="Team Selection Required"
                    description="As a team admin, you need to select your team first before adding models."
                    type="info"
                    showIcon
                    className="mb-4"
                  />
                )}
              </>
            )}
            {(isAdmin || (isTeamAdmin && teamAdminSelectedTeam)) && (
              <>
                <Form.Item
                  rules={[{ required: true, message: "Required" }]}
                  label="Provider:"
                  name="custom_llm_provider"
                  tooltip="E.g. OpenAI, Azure OpenAI, Anthropic, Bedrock, etc."
                  labelCol={{ span: 10 }}
                  labelAlign="left"
                >
                  <AntdSelect
                    virtual={false}
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
                            <ProviderLogo provider={providerKey} className="w-5 h-5" />
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
                    filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
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
                          {isDeviceCodeProvider ? (
                            renderGhDeviceCodeFlow()
                          ) : (
                            <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />
                          )}
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
                {/* Team-only Model Switch - Only show for proxy admins, not team admins */}
                {(isAdmin || !isTeamAdmin) && (
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
                )}

                {/* Conditional Team Selection */}
                {isTeamOnly && (isAdmin || !isTeamAdmin) && (
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
                  guardrailsList={guardrailsList || []}
                  tagsList={tagsList || {}}
                  accessToken={accessToken || ""}
                />
              </>
            )}
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

export default AddModelForm;
