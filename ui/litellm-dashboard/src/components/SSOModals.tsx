import React, { useEffect, useState } from "react";
import { Modal, Form, Input, Button as Button2, Select, Checkbox } from "antd";
import { Text, TextInput } from "@tremor/react";
import { getSSOSettings, updateSSOSettings } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
import { parseErrorMessage } from "./shared/errorUtils";
import { useTranslation } from "react-i18next";

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

const fieldLabelKeys: Record<string, string> = {
  google_client_id: "ssoModals.fieldLabels.googleClientId",
  google_client_secret: "ssoModals.fieldLabels.googleClientSecret",
  microsoft_client_id: "ssoModals.fieldLabels.microsoftClientId",
  microsoft_client_secret: "ssoModals.fieldLabels.microsoftClientSecret",
  microsoft_tenant: "ssoModals.fieldLabels.microsoftTenant",
  generic_client_id: "ssoModals.fieldLabels.genericClientId",
  generic_client_secret: "ssoModals.fieldLabels.genericClientSecret",
  generic_authorization_endpoint: "ssoModals.fieldLabels.authorizationEndpoint",
  generic_token_endpoint: "ssoModals.fieldLabels.tokenEndpoint",
  generic_userinfo_endpoint: "ssoModals.fieldLabels.userinfoEndpoint",
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
  const { t } = useTranslation();
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

            // Extract role mappings if they exist
            let roleMappingFields = {};
            if (ssoData.values.role_mappings) {
              const roleMappings = ssoData.values.role_mappings;

              // Helper function to join arrays into comma-separated strings
              const joinTeams = (teams: string[] | undefined): string => {
                if (!teams || teams.length === 0) return "";
                return teams.join(", ");
              };

              roleMappingFields = {
                use_role_mappings: true,
                group_claim: roleMappings.group_claim,
                default_role: roleMappings.default_role || "internal_user",
                proxy_admin_teams: joinTeams(roleMappings.roles?.proxy_admin),
                admin_viewer_teams: joinTeams(roleMappings.roles?.proxy_admin_viewer),
                internal_user_teams: joinTeams(roleMappings.roles?.internal_user),
                internal_viewer_teams: joinTeams(roleMappings.roles?.internal_user_viewer),
              };
            }

            // Set form values with existing data (excluding UI access control fields)
            const formValues = {
              sso_provider: selectedProvider,
              proxy_base_url: ssoData.values.proxy_base_url,
              user_email: ssoData.values.user_email,
              ...ssoData.values,
              ...roleMappingFields,
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
      NotificationsManager.fromBackend(t("ssoModals.noAccessToken"));
      return;
    }

    try {
      const {
        proxy_admin_teams,
        admin_viewer_teams,
        internal_user_teams,
        internal_viewer_teams,
        default_role,
        group_claim,
        use_role_mappings,
        ...rest
      } = formValues;

      const payload: any = {
        ...rest,
      };

      // Add role mappings if use_role_mappings is checked
      if (use_role_mappings) {
        // Helper function to split comma-separated string into array
        const splitTeams = (teams: string | undefined): string[] => {
          if (!teams || teams.trim() === "") return [];
          return teams
            .split(",")
            .map((team) => team.trim())
            .filter((team) => team.length > 0);
        };

        // Map default role display values to backend values
        const defaultRoleMapping: Record<string, string> = {
          internal_user_viewer: "internal_user_viewer",
          internal_user: "internal_user",
          proxy_admin_viewer: "proxy_admin_viewer",
          proxy_admin: "proxy_admin",
        };

        payload.role_mappings = {
          provider: "generic",
          group_claim,
          default_role: defaultRoleMapping[default_role] || "internal_user",
          roles: {
            proxy_admin: splitTeams(proxy_admin_teams),
            proxy_admin_viewer: splitTeams(admin_viewer_teams),
            internal_user: splitTeams(internal_user_teams),
            internal_user_viewer: splitTeams(internal_viewer_teams),
          },
        };
      }

      // Save SSO settings using the new API
      await updateSSOSettings(accessToken, payload);

      // Continue with the original flow (show instructions)
      handleShowInstructions(formValues);
    } catch (error: unknown) {
      NotificationsManager.fromBackend(t("ssoModals.saveSettingsFailed", { error: parseErrorMessage(error) }));
    }
  };

  // Handle clearing SSO settings
  const handleClearSSO = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend(t("ssoModals.noAccessToken"));
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
        role_mappings: null,
      };

      await updateSSOSettings(accessToken, clearSettings);

      // Clear the form
      form.resetFields();

      // Close the confirmation modal
      setIsClearConfirmModalVisible(false);

      // Close the main SSO modal and trigger refresh
      handleAddSSOOk();

      NotificationsManager.success(t("ssoModals.clearSettingsSuccess"));
    } catch (error) {
      console.error("Failed to clear SSO settings:", error);
      NotificationsManager.fromBackend(t("ssoModals.clearSettingsFailed"));
    }
  };

  // Helper function to render provider fields
  const renderProviderFields = (provider: string) => {
    const config = ssoProviderConfigs[provider];
    if (!config) return null;

    return config.fields.map((field) => {
      const labelKey = fieldLabelKeys[field.name];
      const label = labelKey ? t(labelKey) : field.label;
      return (
        <Form.Item
          key={field.name}
          label={label}
          name={field.name}
          rules={[{ required: true, message: t("ssoModals.fieldRequired", { label: label.toLowerCase() }) }]}
        >
          {field.name.includes("client") ? <Input.Password /> : <TextInput placeholder={field.placeholder} />}
        </Form.Item>
      );
    });
  };

  return (
    <>
      <Modal
        title={ssoConfigured ? t("ssoModals.editTitle") : t("ssoModals.addTitle")}
        open={isAddSSOModalVisible}
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
              label={t("ssoModals.ssoProviderLabel")}
              name="sso_provider"
              rules={[{ required: true, message: t("ssoModals.ssoProviderRequired") }]}
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
              label={t("ssoModals.proxyAdminEmailLabel")}
              name="user_email"
              rules={[{ required: true, message: t("ssoModals.proxyAdminEmailRequired") }]}
            >
              <TextInput />
            </Form.Item>
            <Form.Item
              label={t("ssoModals.proxyBaseUrlLabel")}
              name="proxy_base_url"
              normalize={(value) => value?.trim()}
              rules={[
                { required: true, message: t("ssoModals.proxyBaseUrlRequired") },
                {
                  pattern: /^https?:\/\/.+/,
                  message: t("ssoModals.proxyBaseUrlInvalidScheme"),
                },
                {
                  validator: (_, value) => {
                    // Only check for trailing slash if the URL starts with http:// or https://
                    if (value && /^https?:\/\/.+/.test(value) && value.endsWith("/")) {
                      return Promise.reject(t("ssoModals.proxyBaseUrlTrailingSlash"));
                    }
                    return Promise.resolve();
                  },
                },
              ]}
            >
              <TextInput placeholder="https://example.com" />
            </Form.Item>

            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) => prevValues.sso_provider !== currentValues.sso_provider}
            >
              {({ getFieldValue }) => {
                const provider = getFieldValue("sso_provider");
                return provider === "okta" || provider === "generic" ? (
                  <Form.Item
                    label={t("ssoModals.useRoleMappingsLabel")}
                    name="use_role_mappings"
                    valuePropName="checked"
                  >
                    <Checkbox />
                  </Form.Item>
                ) : null;
              }}
            </Form.Item>

            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) =>
                prevValues.use_role_mappings !== currentValues.use_role_mappings
              }
            >
              {({ getFieldValue }) => {
                const useRoleMappings = getFieldValue("use_role_mappings");
                return useRoleMappings ? (
                  <Form.Item
                    label={t("ssoModals.groupClaimLabel")}
                    name="group_claim"
                    rules={[{ required: true, message: t("ssoModals.groupClaimRequired") }]}
                  >
                    <TextInput />
                  </Form.Item>
                ) : null;
              }}
            </Form.Item>

            <Form.Item
              noStyle
              shouldUpdate={(prevValues, currentValues) =>
                prevValues.use_role_mappings !== currentValues.use_role_mappings
              }
            >
              {({ getFieldValue }) => {
                const useRoleMappings = getFieldValue("use_role_mappings");
                return useRoleMappings ? (
                  <>
                    <Form.Item label={t("ssoModals.defaultRoleLabel")} name="default_role" initialValue="Internal User">
                      <Select>
                        <Select.Option value="internal_user_viewer">{t("ssoModals.roleInternalViewer")}</Select.Option>
                        <Select.Option value="internal_user">{t("ssoModals.roleInternalUser")}</Select.Option>
                        <Select.Option value="proxy_admin_viewer">{t("ssoModals.roleAdminViewer")}</Select.Option>
                        <Select.Option value="proxy_admin">{t("ssoModals.roleProxyAdmin")}</Select.Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label={t("ssoModals.proxyAdminTeamsLabel")} name="proxy_admin_teams">
                      <TextInput />
                    </Form.Item>

                    <Form.Item label={t("ssoModals.adminViewerTeamsLabel")} name="admin_viewer_teams">
                      <TextInput />
                    </Form.Item>

                    <Form.Item label={t("ssoModals.internalUserTeamsLabel")} name="internal_user_teams">
                      <TextInput />
                    </Form.Item>

                    <Form.Item label={t("ssoModals.internalViewerTeamsLabel")} name="internal_viewer_teams">
                      <TextInput />
                    </Form.Item>
                  </>
                ) : null;
              }}
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
                {t("common.clear")}
              </Button2>
            )}
            <Button2 htmlType="submit">{t("common.save")}</Button2>
          </div>
        </Form>
      </Modal>

      {/* Clear Confirmation Modal */}
      <Modal
        title={t("ssoModals.clearConfirmTitle")}
        open={isClearConfirmModalVisible}
        onOk={handleClearSSO}
        onCancel={() => setIsClearConfirmModalVisible(false)}
        okText={t("ssoModals.clearConfirmOk")}
        cancelText={t("common.cancel")}
        okButtonProps={{
          danger: true,
          style: {
            backgroundColor: "#dc2626",
            borderColor: "#dc2626",
          },
        }}
      >
        <p>{t("ssoModals.clearConfirmBody1")}</p>
        <p>{t("ssoModals.clearConfirmBody2")}</p>
      </Modal>

      <Modal
        title={t("ssoModals.instructionsTitle")}
        open={isInstructionsModalVisible}
        width={800}
        footer={null}
        onOk={handleInstructionsOk}
        onCancel={handleInstructionsCancel}
      >
        <p>{t("ssoModals.instructionsIntro")}</p>
        <Text className="mt-2">{t("ssoModals.instructionStep1")}</Text>
        <Text className="mt-2">{t("ssoModals.instructionStep2")}</Text>
        <Text className="mt-2">{t("ssoModals.instructionStep3")}</Text>
        <Text className="mt-2">{t("ssoModals.instructionStep4")}</Text>
        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button2 onClick={handleInstructionsOk}>{t("common.done")}</Button2>
        </div>
      </Modal>
    </>
  );
};

export { ssoProviderConfigs }; // Export for use in other components
export default SSOModals;
