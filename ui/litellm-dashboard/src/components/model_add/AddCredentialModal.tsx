import { TextInput } from "@tremor/react";
import { Select as AntdSelect, Button, Form, Modal, Spin, Tooltip, Typography } from "antd";
import type { UploadProps } from "antd/es/upload";
import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  credentialCreateCall,
  githubCopilotInitiateAuth,
  githubCopilotCheckStatus,
  chatgptInitiateAuth,
  chatgptCheckStatus,
} from "@/components/networking";
import NotificationsManager from "../molecules/notifications_manager";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { Providers, providerLogoMap } from "../provider_info_helpers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
const { Link, Text } = Typography;

interface AddCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  onAddCredential: (values: any) => void;
  uploadProps: UploadProps;
  initialCredentialName?: string;
  initialProvider?: string;
}


const AddCredentialsModal: React.FC<AddCredentialsModalProps> = ({ open, onCancel, onAddCredential, uploadProps, initialCredentialName, initialProvider }) => {
  const [form] = Form.useForm();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.OpenAI);
  const { accessToken } = useAuthorized();
  const { data: providerMetadata } = useProviderFields();

  // Device code flow state
  const [deviceCodeState, setDeviceCodeState] = useState<
    | { phase: "idle" }
    | { phase: "polling"; deviceCode: string; userCode: string; verificationUri: string }
    | { phase: "success"; credentialName: string }
    | { phase: "error"; message: string }
  >({ phase: "idle" });
  // Hold access_token in a ref — never rendered, never put in form fields
  const accessTokenRef = useRef<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Determine if the selected provider uses device_code auth flow and get its litellm_provider
  const deviceCodeProviderInfo = React.useMemo(() => {
    if (!providerMetadata) return null;
    const info = providerMetadata.find(
      (p) =>
        p.provider === selectedProvider ||
        p.provider_display_name === Providers[selectedProvider as keyof typeof Providers],
    );
    if (info?.auth_flow === "device_code") return info;
    return null;
  }, [selectedProvider, providerMetadata]);
  const isDeviceCodeProvider = deviceCodeProviderInfo != null;

  // Cleanup polling on unmount or modal close
  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  useEffect(() => {
    return stopPolling;
  }, [stopPolling]);

  const handleCancel = () => {
    stopPolling();
    setDeviceCodeState({ phase: "idle" });
    accessTokenRef.current = null;
    onCancel();
    form.resetFields();
  };

  const handleSubmit = (values: any) => {
    const filteredValues = Object.entries(values).reduce((acc, [key, value]) => {
      if (value !== "" && value !== undefined && value !== null) {
        acc[key] = value;
      }
      return acc;
    }, {} as any);
    onAddCredential(filteredValues);
    form.resetFields();
  };

  const handleStartDeviceCode = async () => {
    const credentialName = form.getFieldValue("credential_name");
    if (!credentialName) {
      form.validateFields(["credential_name"]);
      return;
    }
    if (!accessToken || !deviceCodeProviderInfo) return;

    const litellmProvider = deviceCodeProviderInfo.litellm_provider;
    const providerLabel = deviceCodeProviderInfo.provider_display_name || litellmProvider;

    try {
      // Initiate — dispatch to the right provider
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
        // Default: GitHub Copilot
        const result = await githubCopilotInitiateAuth(accessToken);
        deviceId = result.device_code;
        userCode = result.user_code;
        verificationUri = result.verification_uri;
        pollIntervalMs = result.poll_interval_ms;
      }

      setDeviceCodeState({
        phase: "polling",
        deviceCode: deviceId,
        userCode,
        verificationUri,
      });

      if (!pollIntervalMs) throw new Error(`${providerLabel} initiate response missing poll_interval_ms`);
      // Mutable baseline — ratchets up on slow_down so that subsequent
      // normal "pending" responses keep using the increased interval.
      let currentPollInterval = pollIntervalMs;

      // setTimeout-based loop so each poll fires only after the previous one
      // completes, and slow_down / rate-limit intervals are respected exactly.
      const schedulePoll = (delayMs: number) => {
        pollingRef.current = setTimeout(async () => {
          try {
            let apiKey: string | undefined;
            let failed = false;
            let errorMsg: string | undefined;
            let retryAfterMs: number | undefined;

            if (litellmProvider === "chatgpt") {
              const status = await chatgptCheckStatus(accessToken, deviceId, userCode);
              console.log(`[${providerLabel} AddCredential] poll response:`, status);
              if (status.status === "complete" && status.refresh_token) {
                apiKey = status.refresh_token;
              } else if (status.status === "failed") {
                failed = true;
                errorMsg = status.error;
              }
            } else {
              const status = await githubCopilotCheckStatus(accessToken, deviceId);
              console.log(`[${providerLabel} AddCredential] poll response:`, status);
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
              stopPolling();
              accessTokenRef.current = apiKey;
              try {
                await credentialCreateCall(accessToken, {
                  credential_name: credentialName,
                  credential_values: { api_key: apiKey },
                  credential_info: { custom_llm_provider: litellmProvider },
                });
                setDeviceCodeState({ phase: "success", credentialName });
              } catch (e) {
                console.error(`[${providerLabel} AddCredential] credentialCreateCall failed:`, e);
                NotificationsManager.error(
                  `Failed to save credential: ${e instanceof Error ? e.message : "Unknown error"}`,
                );
                setDeviceCodeState({ phase: "error", message: "Failed to save credential" });
              }
            } else if (failed) {
              stopPolling();
              setDeviceCodeState({ phase: "error", message: errorMsg || "Authorization failed" });
            } else {
              // pending — ratchet up the baseline if provider requested slower
              if (retryAfterMs != null) {
                currentPollInterval = retryAfterMs;
              }
              schedulePoll(currentPollInterval);
            }
          } catch (e) {
            console.error(`[${providerLabel} AddCredential] poll error:`, e);
            stopPolling();
            setDeviceCodeState({ phase: "error", message: "Failed to check authorization status" });
          }
        }, delayMs);
      };
      schedulePoll(currentPollInterval);
    } catch {
      setDeviceCodeState({ phase: "error", message: `Failed to start ${providerLabel} authorization` });
    }
  };

  const handleSuccessClose = () => {
    stopPolling();
    setDeviceCodeState({ phase: "idle" });
    accessTokenRef.current = null;
    onCancel();
    form.resetFields();
  };

  const providerDisplayName = deviceCodeProviderInfo?.provider_display_name || "Provider";

  const renderDeviceCodeFlow = () => {
    switch (deviceCodeState.phase) {
      case "idle":
        return (
          <div className="text-center py-4">
            <Text className="block mb-4">
              {providerDisplayName} uses OAuth Device Code authorization. Click below to start.
            </Text>
            <Button type="primary" onClick={handleStartDeviceCode}>
              Start {providerDisplayName} Authorization
            </Button>
          </div>
        );
      case "polling":
        return (
          <div className="text-center py-4">
            <Text className="block mb-2">
              Enter this code to authorize:
            </Text>
            <div
              style={{
                fontSize: "2rem",
                fontWeight: "bold",
                fontFamily: "monospace",
                letterSpacing: "0.3em",
                margin: "16px 0",
                padding: "12px 24px",
                background: "#f5f5f5",
                borderRadius: 8,
                display: "inline-block",
                userSelect: "all",
              }}
            >
              {deviceCodeState.userCode}
            </div>
            <div className="mb-4">
              <Button
                type="link"
                onClick={() => window.open(deviceCodeState.verificationUri, "_blank")}
              >
                Open {deviceCodeState.verificationUri}
              </Button>
            </div>
            <Spin />
            <Text className="block mt-2 mb-4" type="secondary">
              Waiting for {providerDisplayName} authorization...
            </Text>
            <Button onClick={handleCancel}>Cancel</Button>
          </div>
        );
      case "success":
        return (
          <div className="text-center py-4">
            <Text className="block mb-4" type="success" style={{ fontSize: "1.1rem" }}>
              {providerDisplayName} credential &quot;{deviceCodeState.credentialName}&quot; created successfully!
            </Text>
            <Button type="primary" onClick={handleSuccessClose}>
              Done
            </Button>
          </div>
        );
      case "error":
        return (
          <div className="text-center py-4">
            <Text className="block mb-4" type="danger">
              {deviceCodeState.message}
            </Text>
            <Button
              onClick={() => setDeviceCodeState({ phase: "idle" })}
              style={{ marginRight: 8 }}
            >
              Retry
            </Button>
            <Button onClick={handleCancel}>Cancel</Button>
          </div>
        );
    }
  };

  return (
    <Modal
      title="Add New Credential"
      open={open}
      onCancel={handleCancel}
      footer={null}
      width={600}
    >
      <Form form={form} onFinish={handleSubmit} layout="vertical">
        {/* Credential Name */}
        <Form.Item
          label="Credential Name:"
          name="credential_name"
          rules={[{ required: true, message: "Credential name is required" }]}
        >
          <TextInput placeholder="Enter a friendly name for these credentials" />
        </Form.Item>

        {/* Provider Selection */}
        <Form.Item
          rules={[{ required: true, message: "Required" }]}
          label="Provider:"
          name="custom_llm_provider"
          tooltip="Helper to auto-populate provider specific fields"
        >
          <AntdSelect
            showSearch
            onChange={(value) => {
              setSelectedProvider(value as Providers);
              form.setFieldValue("custom_llm_provider", value);
              // Reset device code state when provider changes
              stopPolling();
              setDeviceCodeState({ phase: "idle" });
              accessTokenRef.current = null;
            }}
          >
            {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => (
              <AntdSelect.Option key={providerEnum} value={providerEnum}>
                <div className="flex items-center space-x-2">
                  <img
                    src={providerLogoMap[providerDisplayName]}
                    alt={`${providerEnum} logo`}
                    className="w-5 h-5"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      const parent = target.parentElement;
                      if (parent) {
                        const fallbackDiv = document.createElement("div");
                        fallbackDiv.className =
                          "w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs";
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

        {isDeviceCodeProvider ? (
          renderDeviceCodeFlow()
        ) : (
          <>
            <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />

            {/* Modal Footer */}
            <div className="flex justify-between items-center">
              <Tooltip title="Get help on our github">
                <Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Link>
              </Tooltip>

              <div>
                <Button
                  onClick={handleCancel}
                  style={{ marginRight: 10 }}
                >
                  Cancel
                </Button>
                <Button htmlType="submit">{"Add Credential"}</Button>
              </div>
            </div>
          </>
        )}
      </Form>
    </Modal>
  );
};

export default AddCredentialsModal;
