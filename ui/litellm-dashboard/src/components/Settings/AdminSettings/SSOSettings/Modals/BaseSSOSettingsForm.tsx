"use client";

import { TextInput } from "@tremor/react";
import { Checkbox, Form, Input, Select } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";
import { ssoProviderLogoMap, ssoProviderDisplayNames } from "../constants";

type TranslateFn = (key: string, options?: Record<string, any>) => string;

export interface BaseSSOSettingsFormProps {
  form: any; // Replace with proper Form type if available
  onFormSubmit: (formValues: Record<string, any>) => Promise<void>;
}

// Define the SSO provider configuration type
export interface SSOProviderConfig {
  envVarMap: Record<string, string>;
  fields: Array<{
    labelKey: string;
    name: string;
    placeholder?: string;
  }>;
}

// Define configurations for each SSO provider
export const getSSOProviderConfigs = (t: TranslateFn): Record<string, SSOProviderConfig> => ({
  google: {
    envVarMap: {
      google_client_id: "GOOGLE_CLIENT_ID",
      google_client_secret: "GOOGLE_CLIENT_SECRET",
    },
    fields: [
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.googleClientId"),
        name: "google_client_id",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.googleClientSecret"),
        name: "google_client_secret",
      },
    ],
  },
  microsoft: {
    envVarMap: {
      microsoft_client_id: "MICROSOFT_CLIENT_ID",
      microsoft_client_secret: "MICROSOFT_CLIENT_SECRET",
      microsoft_tenant: "MICROSOFT_TENANT",
    },
    fields: [
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.microsoftClientId"),
        name: "microsoft_client_id",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.microsoftClientSecret"),
        name: "microsoft_client_secret",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.microsoftTenant"),
        name: "microsoft_tenant",
      },
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
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.genericClientId"),
        name: "generic_client_id",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.genericClientSecret"),
        name: "generic_client_secret",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.authorizationEndpoint"),
        name: "generic_authorization_endpoint",
        placeholder: "https://your-domain/authorize",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.tokenEndpoint"),
        name: "generic_token_endpoint",
        placeholder: "https://your-domain/token",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.userinfoEndpoint"),
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
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.genericClientId"),
        name: "generic_client_id",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.genericClientSecret"),
        name: "generic_client_secret",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.authorizationEndpoint"),
        name: "generic_authorization_endpoint",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.tokenEndpoint"),
        name: "generic_token_endpoint",
      },
      {
        labelKey: t("settingsPages.baseSSOSettingsForm.fieldLabels.userinfoEndpoint"),
        name: "generic_userinfo_endpoint",
      },
    ],
  },
});

// Helper function to render provider fields
export const renderProviderFields = (provider: string, t: TranslateFn = (k: string) => k as any) => {
  const configs = getSSOProviderConfigs(t);
  const config = configs[provider];
  if (!config) return null;

  return config.fields.map((field) => (
    <Form.Item
      key={field.name}
      label={field.labelKey}
      name={field.name}
      rules={[
        {
          required: true,
          message: t("settingsPages.baseSSOSettingsForm.fieldRequired", { label: field.labelKey.toLowerCase() }),
        },
      ]}
    >
      {field.name.includes("client") ? <Input.Password /> : <TextInput placeholder={field.placeholder} />}
    </Form.Item>
  ));
};

const BaseSSOSettingsForm: React.FC<BaseSSOSettingsFormProps> = ({ form, onFormSubmit }) => {
  const { t } = useTranslation();
  return (
    <div>
      <Form form={form} onFinish={onFormSubmit} labelCol={{ span: 8 }} wrapperCol={{ span: 16 }} labelAlign="left">
        <Form.Item
          label={t("settingsPages.baseSSOSettingsForm.ssoProviderLabel")}
          name="sso_provider"
          rules={[{ required: true, message: t("settingsPages.baseSSOSettingsForm.ssoProviderRequired") }]}
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
                    {ssoProviderDisplayNames[value] || value.charAt(0).toUpperCase() + value.slice(1) + " SSO"}
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
            return provider ? renderProviderFields(provider, t) : null;
          }}
        </Form.Item>

        <Form.Item
          label={t("settingsPages.baseSSOSettingsForm.proxyAdminEmailLabel")}
          name="user_email"
          rules={[{ required: true, message: t("settingsPages.baseSSOSettingsForm.proxyAdminEmailRequired") }]}
        >
          <TextInput />
        </Form.Item>
        <Form.Item
          label={t("settingsPages.baseSSOSettingsForm.proxyBaseUrlLabel")}
          name="proxy_base_url"
          normalize={(value) => value?.trim()}
          rules={[
            { required: true, message: t("settingsPages.baseSSOSettingsForm.proxyBaseUrlRequired") },
            {
              pattern: /^https?:\/\/.+/,
              message: t("settingsPages.baseSSOSettingsForm.proxyBaseUrlInvalidScheme"),
            },
            {
              validator: (_, value) => {
                // Only check for trailing slash if the URL starts with http:// or https://
                if (value && /^https?:\/\/.+/.test(value) && value.endsWith("/")) {
                  return Promise.reject(t("settingsPages.baseSSOSettingsForm.proxyBaseUrlTrailingSlash"));
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
                label={t("settingsPages.baseSSOSettingsForm.useRoleMappingsLabel")}
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
            prevValues.use_role_mappings !== currentValues.use_role_mappings ||
            prevValues.sso_provider !== currentValues.sso_provider
          }
        >
          {({ getFieldValue }) => {
            const useRoleMappings = getFieldValue("use_role_mappings");
            const provider = getFieldValue("sso_provider");
            const supportsRoleMappings = provider === "okta" || provider === "generic";
            return useRoleMappings && supportsRoleMappings ? (
              <Form.Item
                label={t("settingsPages.baseSSOSettingsForm.groupClaimLabel")}
                name="group_claim"
                rules={[{ required: true, message: t("settingsPages.baseSSOSettingsForm.groupClaimRequired") }]}
              >
                <TextInput />
              </Form.Item>
            ) : null;
          }}
        </Form.Item>

        <Form.Item
          noStyle
          shouldUpdate={(prevValues, currentValues) =>
            prevValues.use_role_mappings !== currentValues.use_role_mappings ||
            prevValues.sso_provider !== currentValues.sso_provider
          }
        >
          {({ getFieldValue }) => {
            const useRoleMappings = getFieldValue("use_role_mappings");
            const provider = getFieldValue("sso_provider");
            const supportsRoleMappings = provider === "okta" || provider === "generic";
            return useRoleMappings && supportsRoleMappings ? (
              <>
                <Form.Item
                  label={t("settingsPages.baseSSOSettingsForm.defaultRoleLabel")}
                  name="default_role"
                  initialValue="Internal User"
                >
                  <Select>
                    <Select.Option value="internal_user_viewer">
                      {t("settingsPages.baseSSOSettingsForm.roleInternalViewer")}
                    </Select.Option>
                    <Select.Option value="internal_user">
                      {t("settingsPages.baseSSOSettingsForm.roleInternalUser")}
                    </Select.Option>
                    <Select.Option value="proxy_admin_viewer">
                      {t("settingsPages.baseSSOSettingsForm.roleAdminViewer")}
                    </Select.Option>
                    <Select.Option value="proxy_admin">
                      {t("settingsPages.baseSSOSettingsForm.roleProxyAdmin")}
                    </Select.Option>
                  </Select>
                </Form.Item>

                <Form.Item label={t("settingsPages.baseSSOSettingsForm.proxyAdminTeamsLabel")} name="proxy_admin_teams">
                  <TextInput />
                </Form.Item>

                <Form.Item
                  label={t("settingsPages.baseSSOSettingsForm.adminViewerTeamsLabel")}
                  name="admin_viewer_teams"
                >
                  <TextInput />
                </Form.Item>

                <Form.Item
                  label={t("settingsPages.baseSSOSettingsForm.internalUserTeamsLabel")}
                  name="internal_user_teams"
                >
                  <TextInput />
                </Form.Item>

                <Form.Item
                  label={t("settingsPages.baseSSOSettingsForm.internalViewerTeamsLabel")}
                  name="internal_viewer_teams"
                >
                  <TextInput />
                </Form.Item>
              </>
            ) : null;
          }}
        </Form.Item>

        <Form.Item
          noStyle
          shouldUpdate={(prevValues, currentValues) => prevValues.sso_provider !== currentValues.sso_provider}
        >
          {({ getFieldValue }) => {
            const provider = getFieldValue("sso_provider");
            return provider === "okta" || provider === "generic" ? (
              <Form.Item
                label={t("settingsPages.baseSSOSettingsForm.useTeamMappingsLabel")}
                name="use_team_mappings"
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
            prevValues.use_team_mappings !== currentValues.use_team_mappings ||
            prevValues.sso_provider !== currentValues.sso_provider
          }
        >
          {({ getFieldValue }) => {
            const useTeamMappings = getFieldValue("use_team_mappings");
            const provider = getFieldValue("sso_provider");
            const supportsTeamMappings = provider === "okta" || provider === "generic";
            return useTeamMappings && supportsTeamMappings ? (
              <Form.Item
                label={t("settingsPages.baseSSOSettingsForm.teamIdsJwtFieldLabel")}
                name="team_ids_jwt_field"
                rules={[{ required: true, message: t("settingsPages.baseSSOSettingsForm.teamIdsJwtFieldRequired") }]}
              >
                <TextInput />
              </Form.Item>
            ) : null;
          }}
        </Form.Item>
      </Form>
    </div>
  );
};

export default BaseSSOSettingsForm;
