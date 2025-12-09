import React, { useEffect, useState } from "react";
import { Modal, Form, Input, Button as Button2, Select } from "antd";
import { Text, TextInput } from "@tremor/react";
import { getSSOSettings, updateSSOSettings } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
import { parseErrorMessage } from "./shared/errorUtils";

interface SSOModalsProps {
  isAddSSOModalVisible: boolean;
  isInstructionsModalVisible: boolean;
  handleAddSSOOk: () => void;
  handleAddSSOCancel: () => void;
  handleShowInstructions: (formValues: Record<string, any>) => void;
  handleInstructionsOk: () => void;
  handleInstructionsCancel: () => void;
  form: any; // Replace with proper Form type if available
  accessToken: string | null;
  ssoConfigured?: boolean; // Add optional prop to indicate if SSO is configured
}

const ssoProviderLogoMap: Record<string, string> = {
  google: "https://artificialanalysis.ai/img/logos/google_small.svg",
  microsoft: "https://upload.wikimedia.org/wikipedia/commons/a/a8/Microsoft_Azure_Logo.svg",
  okta: "https://www.okta.com/sites/default/files/Okta_Logo_BrightBlue_Medium.png",
  generic: "",
};

// Define the SSO provider configuration type
interface SSOProviderConfig {
  envVarMap: Record<string, string>;
  fields: Array<{
    label: string;
    name: string;
    placeholder?: string;
  }>;
}

// Define configurations for each SSO provider
const ssoProviderConfigs: Record<string, SSOProviderConfig> = {
  google: {
    envVarMap: {
      google_client_id: "GOOGLE_CLIENT_ID",
      google_client_secret: "GOOGLE_CLIENT_SECRET",
    },
    fields: [
      { label: "Google Client ID", name: "google_client_id" },
      { label: "Google Client Secret", name: "google_client_secret" },
    ],
  },
  microsoft: {
    envVarMap: {
      microsoft_client_id: "MICROSOFT_CLIENT_ID",
      microsoft_client_secret: "MICROSOFT_CLIENT_SECRET",
      microsoft_tenant: "MICROSOFT_TENANT",
    },
    fields: [
      { label: "Microsoft Client ID", name: "microsoft_client_id" },
      { label: "Microsoft Client Secret", name: "microsoft_client_secret" },
      { label: "Microsoft Tenant", name: "microsoft_tenant" },
    ],
  },
  okta: {
    envVarMap: {
      generic_client_id: "GENERIC_CLIENT_ID",
      generic_client_secret: "GENERIC_CLIENT_SECRET",
      generic_authorization_endpoint: "GENERIC_AUTHORIZATION_ENDPOINT",
      generic_token_endpoint: "GENERIC_TOKEN_ENDPOINT",
      generic_userinfo_endpoint: "GENERIC_USERINFO_ENDPOINT",
    },
    fields: [
      { label: "Generic Client ID", name: "generic_client_id" },
      { label: "Generic Client Secret", name: "generic_client_secret" },
      {
        label: "Authorization Endpoint",
        name: "generic_authorization_endpoint",
        placeholder: "https://your-domain/authorize",
      },
      { label: "Token Endpoint", name: "generic_token_endpoint", placeholder: "https://your-domain/token" },
      {
        label: "Userinfo Endpoint",
        name: "generic_userinfo_endpoint",
        placeholder: "https://your-domain/userinfo",
      },
    ],
  },
  generic: {
    envVarMap: {
      generic_client_id: "GENERIC_CLIENT_ID",
      generic_client_secret: "GENERIC_CLIENT_SECRET",
      generic_authorization_endpoint: "GENERIC_AUTHORIZATION_ENDPOINT",
      generic_token_endpoint: "GENERIC_TOKEN_ENDPOINT",
      generic_userinfo_endpoint: "GENERIC_USERINFO_ENDPOINT",
    },
    fields: [
      { label: "Generic Client ID", name: "generic_client_id" },
      { label: "Generic Client Secret", name: "generic_client_secret" },
      { label: "Authorization Endpoint", name: "generic_authorization_endpoint" },
      { label: "Token Endpoint", name: "generic_token_endpoint" },
      { label: "Userinfo Endpoint", name: "generic_userinfo_endpoint" },
    ],
  },
};

const SSOModals: React.FC<SSOModalsProps> = ({
  isAddSSOModalVisible,
  isInstructionsModalVisible,
  handleAddSSOOk,
  handleAddSSOCancel,
  handleShowInstructions,
  handleInstructionsOk,
  handleInstructionsCancel,
  form,
  accessToken,
  ssoConfigured = false, // Default to false if not provided
}) => {
  const [isClearConfirmModalVisible, setIsClearConfirmModalVisible] = useState(false);

  // Load existing SSO settings when modal opens
  useEffect(() => {
    const loadSSOSettings = async () => {
      if (isAddSSOModalVisible && accessToken) {
        try {
          const ssoData = await getSSOSettings(accessToken);
          console.log("Raw SSO data received:", ssoData); // Debug log
          if (ssoData && ssoData.values) {
            console.log("SSO values:", ssoData.values); // Debug log
            console.log("user_email from API:", ssoData.values.user_email); // Debug log

            // Determine which SSO provider is configured
            let selectedProvider = null;
            if (ssoData.values.google_client_id) {
              selectedProvider = "google";
            } else if (ssoData.values.microsoft_client_id) {
              selectedProvider = "microsoft";
            } else if (ssoData.values.generic_client_id) {
              // Check if it looks like Okta based on endpoints
              if (
                ssoData.values.generic_authorization_endpoint?.includes("okta") ||
                ssoData.values.generic_authorization_endpoint?.includes("auth0")
              ) {
                selectedProvider = "okta";
              } else {
                selectedProvider = "generic";
              }
            }

            // Set form values with existing data (excluding UI access control fields)
            const formValues = {
              sso_provider: selectedProvider,
              proxy_base_url: ssoData.values.proxy_base_url,
              user_email: ssoData.values.user_email,
              ...ssoData.values,
            };

            console.log("Setting form values:", formValues); // Debug log

            // Clear form first, then set values with a small delay to ensure proper initialization
            form.resetFields();
            setTimeout(() => {
              form.setFieldsValue(formValues);
              console.log("Form values set, current form values:", form.getFieldsValue()); // Debug log
            }, 100);
          }
        } catch (error) {
          console.error("Failed to load SSO settings:", error);
        }
      }
    };

    loadSSOSettings();
  }, [isAddSSOModalVisible, accessToken, form]);

  // Enhanced form submission handler
  const handleFormSubmit = async (formValues: Record<string, any>) => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    try {
      // Save SSO settings using the new API
      await updateSSOSettings(accessToken, formValues);

      // Continue with the original flow (show instructions)
      handleShowInstructions(formValues);
    } catch (error: unknown) {
      NotificationsManager.fromBackend("Failed to save SSO settings: " + parseErrorMessage(error));
    }
  };

  // Handle clearing SSO settings
  const handleClearSSO = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    try {
      // Clear all SSO settings
      const clearSettings = {
        google_client_id: null,
        google_client_secret: null,
        microsoft_client_id: null,
        microsoft_client_secret: null,
        microsoft_tenant: null,
        generic_client_id: null,
        generic_client_secret: null,
        generic_authorization_endpoint: null,
        generic_token_endpoint: null,
        generic_userinfo_endpoint: null,
        proxy_base_url: null,
        user_email: null,
        sso_provider: null,
      };

      await updateSSOSettings(accessToken, clearSettings);

      // Clear the form
      form.resetFields();

      // Close the confirmation modal
      setIsClearConfirmModalVisible(false);

      // Close the main SSO modal and trigger refresh
      handleAddSSOOk();

      NotificationsManager.success("SSO settings cleared successfully");
    } catch (error) {
      console.error("Failed to clear SSO settings:", error);
      NotificationsManager.fromBackend("Failed to clear SSO settings");
    }
  };

  // Helper function to render provider fields
  const renderProviderFields = (provider: string) => {
    const config = ssoProviderConfigs[provider];
    if (!config) return null;

    return config.fields.map((field) => (
      <Form.Item
        key={field.name}
        label={field.label}
        name={field.name}
        rules={[{ required: true, message: `Please enter the ${field.label.toLowerCase()}` }]}
      >
        {field.name.includes("client") ? <Input.Password /> : <TextInput placeholder={field.placeholder} />}
      </Form.Item>
    ));
  };

  return (
    <>
      <Modal
        title={ssoConfigured ? "Edit SSO Settings" : "Add SSO"}
        visible={isAddSSOModalVisible}
        width={800}
        footer={null}
        onOk={handleAddSSOOk}
        onCancel={handleAddSSOCancel}
      >
        <Form
          form={form}
          onFinish={handleFormSubmit}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            <Form.Item
              label="SSO Provider"
              name="sso_provider"
              rules={[{ required: true, message: "Please select an SSO provider" }]}
            >
              <Select>
                {Object.entries(ssoProviderLogoMap).map(([value, logo]) => (
                  <Select.Option key={value} value={value}>
                    <div style={{ display: "flex", alignItems: "center", padding: "4px 0" }}>
                      {logo && (
                        <img
                          src={logo}
                          alt={value}
                          style={{ height: 24, width: 24, marginRight: 12, objectFit: "contain" }}
                        />
                      )}
                      <span>
                        {value.toLowerCase() === "okta"
                          ? "Okta / Auth0"
                          : value.charAt(0).toUpperCase() + value.slice(1)}{" "}
                        SSO
                      </span>
                    </div>
                  </Select.Option>
                ))}
              </Select>
            </Form.Item>

            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) => prevValues.sso_provider !== currentValues.sso_provider}
            >
              {({ getFieldValue }) => {
                const provider = getFieldValue("sso_provider");
                return provider ? renderProviderFields(provider) : null;
              }}
            </Form.Item>

            <Form.Item
              label="Proxy Admin Email"
              name="user_email"
              rules={[{ required: true, message: "Please enter the email of the proxy admin" }]}
            >
              <TextInput />
            </Form.Item>
            <Form.Item
              label="Proxy Base URL"
              name="proxy_base_url"
              normalize={(value) => value?.trim()}
              rules={[
                { required: true, message: "Please enter the proxy base url" },
                {
                  pattern: /^https?:\/\/.+/,
                  message: "URL must start with http:// or https://",
                },
                {
                  validator: (_, value) => {
                    // Only check for trailing slash if the URL starts with http:// or https://
                    if (value && /^https?:\/\/.+/.test(value) && value.endsWith("/")) {
                      return Promise.reject("URL must not end with a trailing slash");
                    }
                    return Promise.resolve();
                  },
                },
              ]}
            >
              <TextInput placeholder="https://example.com" />
            </Form.Item>
          </>
          <div
            style={{
              textAlign: "right",
              marginTop: "10px",
              display: "flex",
              justifyContent: "flex-end",
              alignItems: "center",
              gap: "8px",
            }}
          >
            {ssoConfigured && (
              <Button2
                onClick={() => setIsClearConfirmModalVisible(true)}
                style={{
                  backgroundColor: "#6366f1",
                  borderColor: "#6366f1",
                  color: "white",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = "#5558eb";
                  e.currentTarget.style.borderColor = "#5558eb";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = "#6366f1";
                  e.currentTarget.style.borderColor = "#6366f1";
                }}
              >
                Clear
              </Button2>
            )}
            <Button2 htmlType="submit">Save</Button2>
          </div>
        </Form>
      </Modal>

      {/* Clear Confirmation Modal */}
      <Modal
        title="Confirm Clear SSO Settings"
        visible={isClearConfirmModalVisible}
        onOk={handleClearSSO}
        onCancel={() => setIsClearConfirmModalVisible(false)}
        okText="Yes, Clear"
        cancelText="Cancel"
        okButtonProps={{
          danger: true,
          style: {
            backgroundColor: "#dc2626",
            borderColor: "#dc2626",
          },
        }}
      >
        <p>Are you sure you want to clear all SSO settings? This action cannot be undone.</p>
        <p>Users will no longer be able to login using SSO after this change.</p>
      </Modal>

      <Modal
        title="SSO Setup Instructions"
        visible={isInstructionsModalVisible}
        width={800}
        footer={null}
        onOk={handleInstructionsOk}
        onCancel={handleInstructionsCancel}
      >
        <p>Follow these steps to complete the SSO setup:</p>
        <Text className="mt-2">1. DO NOT Exit this TAB</Text>
        <Text className="mt-2">2. Open a new tab, visit your proxy base url</Text>
        <Text className="mt-2">3. Confirm your SSO is configured correctly and you can login on the new Tab</Text>
        <Text className="mt-2">4. If Step 3 is successful, you can close this tab</Text>
        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 onClick={handleInstructionsOk}>Done</Button2>
        </div>
      </Modal>
    </>
  );
};

export { ssoProviderConfigs }; // Export for use in other components
export default SSOModals;
