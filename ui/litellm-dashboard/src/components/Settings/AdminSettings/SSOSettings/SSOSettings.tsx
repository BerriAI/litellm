"use client";

import { useSSOSettings, type SSOSettingsValues } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { Button, Card, Descriptions, Space, Tag, Typography } from "antd";
import { Edit, Shield, Trash2 } from "lucide-react";
import { useState } from "react";
import { ssoProviderDisplayNames, ssoProviderLogoMap } from "./constants";
import AddSSOSettingsModal from "./Modals/AddSSOSettingsModal";
import DeleteSSOSettingsModal from "./Modals/DeleteSSOSettingsModal";
import EditSSOSettingsModal from "./Modals/EditSSOSettingsModal";
import RedactableField from "./RedactableField";
import RoleMappings from "./RoleMappings";
import SSOSettingsEmptyPlaceholder from "./SSOSettingsEmptyPlaceholder";
import SSOSettingsLoadingSkeleton from "./SSOSettingsLoadingSkeleton";
import { detectSSOProvider } from "./utils";

const { Title, Text } = Typography;

export default function SSOSettings() {
  const { data: ssoSettings, refetch, isLoading } = useSSOSettings();
  const [isDeleteModalVisible, setIsDeleteModalVisible] = useState(false);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const isSSOConfigured =
    Boolean(ssoSettings?.values.google_client_id) ||
    Boolean(ssoSettings?.values.microsoft_client_id) ||
    Boolean(ssoSettings?.values.generic_client_id);

  const selectedProvider = ssoSettings?.values ? detectSSOProvider(ssoSettings.values) : null;
  const isRoleMappingsEnabled = Boolean(ssoSettings?.values.role_mappings);
  const isTeamMappingsEnabled = Boolean(ssoSettings?.values.team_mappings);

  const renderEndpointValue = (value?: string | null) => (
    <Text className="font-mono text-gray-600 text-sm" copyable={!!value}>
      {value || "-"}
    </Text>
  );

  const renderSimpleValue = (value?: string | null) =>
    value ? value : <span className="text-gray-400 italic">Not configured</span>;

  const renderTeamMappingsField = (values: SSOSettingsValues) => {
    if (!values.team_mappings?.team_ids_jwt_field) {
      return <span className="text-gray-400 italic">Not configured</span>;
    }
    return (
      <Tag>{values.team_mappings.team_ids_jwt_field}</Tag>
    );
  };

  const descriptionsConfig = {
    column: {
      xxl: 1,
      xl: 1,
      lg: 1,
      md: 1,
      sm: 1,
      xs: 1,
    },
  };

  const providerConfigs = {
    google: {
      providerText: ssoProviderDisplayNames.google,
      fields: [
        {
          label: "Client ID",
          render: (values: SSOSettingsValues) => <RedactableField value={values.google_client_id} />,
        },
        {
          label: "Client Secret",
          render: (values: SSOSettingsValues) => <RedactableField value={values.google_client_secret} />,
        },
        { label: "Proxy Base URL", render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url) },
      ],
    },
    microsoft: {
      providerText: ssoProviderDisplayNames.microsoft,
      fields: [
        {
          label: "Client ID",
          render: (values: SSOSettingsValues) => <RedactableField value={values.microsoft_client_id} />,
        },
        {
          label: "Client Secret",
          render: (values: SSOSettingsValues) => <RedactableField value={values.microsoft_client_secret} />,
        },
        { label: "Tenant", render: (values: any) => renderSimpleValue(values.microsoft_tenant) },
        { label: "Proxy Base URL", render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url) },
      ],
    },
    okta: {
      providerText: ssoProviderDisplayNames.okta,
      fields: [
        {
          label: "Client ID",
          render: (values: SSOSettingsValues) => <RedactableField value={values.generic_client_id} />,
        },
        {
          label: "Client Secret",
          render: (values: SSOSettingsValues) => <RedactableField value={values.generic_client_secret} />,
        },
        {
          label: "Authorization Endpoint",
          render: (values: SSOSettingsValues) => renderEndpointValue(values.generic_authorization_endpoint),
        },
        {
          label: "Token Endpoint",
          render: (values: SSOSettingsValues) => renderEndpointValue(values.generic_token_endpoint),
        },
        {
          label: "User Info Endpoint",
          render: (values: SSOSettingsValues) => renderEndpointValue(values.generic_userinfo_endpoint),
        },
        { label: "Proxy Base URL", render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url) },
        isTeamMappingsEnabled ? {
          label: "Team IDs JWT Field",
          render: (values: SSOSettingsValues) => renderTeamMappingsField(values),
        } : null,
      ],
    },
    generic: {
      providerText: ssoProviderDisplayNames.generic,
      fields: [
        {
          label: "Client ID",
          render: (values: SSOSettingsValues) => <RedactableField value={values.generic_client_id} />,
        },
        {
          label: "Client Secret",
          render: (values: SSOSettingsValues) => <RedactableField value={values.generic_client_secret} />,
        },
        {
          label: "Authorization Endpoint",
          render: (values: SSOSettingsValues) => renderEndpointValue(values.generic_authorization_endpoint),
        },
        {
          label: "Token Endpoint",
          render: (values: SSOSettingsValues) => renderEndpointValue(values.generic_token_endpoint),
        },
        {
          label: "User Info Endpoint",
          render: (values: SSOSettingsValues) => renderEndpointValue(values.generic_userinfo_endpoint),
        },
        { label: "Proxy Base URL", render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url) },
        isTeamMappingsEnabled ? {
          label: "Team IDs JWT Field",
          render: (values: SSOSettingsValues) => renderTeamMappingsField(values),
        } : null,
      ],
    },
  };

  const renderSSOSettings = () => {
    if (!ssoSettings?.values || !selectedProvider) return null;

    const { values } = ssoSettings;
    const config = providerConfigs[selectedProvider as keyof typeof providerConfigs];

    if (!config) return null;

    return (
      <Descriptions bordered {...descriptionsConfig}>
        <Descriptions.Item label="Provider">
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            {ssoProviderLogoMap[selectedProvider] && (
              <img
                src={ssoProviderLogoMap[selectedProvider]}
                alt={selectedProvider}
                style={{ height: 24, width: 24, objectFit: "contain" }}
              />
            )}
            <span>{config.providerText}</span>
          </div>
        </Descriptions.Item>
        {config.fields.map((field, index) => field && (
          <Descriptions.Item key={index} label={field.label}>
            {field.render(values)}
          </Descriptions.Item>
        ))}
      </Descriptions>
    );
  };

  return (
    <>
      {isLoading ? (
        <SSOSettingsLoadingSkeleton />
      ) : (
        <Space direction="vertical" size="large" className="w-full">
          <Card>
            <Space direction="vertical" size="large" className="w-full">
              {/* Header Section */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Shield className="w-6 h-6 text-gray-400" />
                  <div>
                    <Title level={3}>SSO Configuration</Title>
                    <Text type="secondary">Manage Single Sign-On authentication settings</Text>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  {isSSOConfigured && (
                    <>
                      <Button icon={<Edit className="w-4 h-4" />} onClick={() => setIsEditModalVisible(true)}>
                        Edit SSO Settings
                      </Button>
                      <Button
                        danger
                        icon={<Trash2 className="w-4 h-4" />}
                        onClick={() => setIsDeleteModalVisible(true)}
                      >
                        Delete SSO Settings
                      </Button>
                    </>
                  )}
                </div>
              </div>

              {isSSOConfigured ? (
                renderSSOSettings()
              ) : (
                <SSOSettingsEmptyPlaceholder onAdd={() => setIsAddModalVisible(true)} />
              )}
            </Space>
          </Card>
          {isRoleMappingsEnabled && <RoleMappings roleMappings={ssoSettings?.values.role_mappings} />}
        </Space>
      )}

      <DeleteSSOSettingsModal
        isVisible={isDeleteModalVisible}
        onCancel={() => setIsDeleteModalVisible(false)}
        onSuccess={() => refetch()}
      />

      <AddSSOSettingsModal
        isVisible={isAddModalVisible}
        onCancel={() => setIsAddModalVisible(false)}
        onSuccess={() => {
          setIsAddModalVisible(false);
          refetch();
        }}
      />

      <EditSSOSettingsModal
        isVisible={isEditModalVisible}
        onCancel={() => setIsEditModalVisible(false)}
        onSuccess={() => {
          setIsEditModalVisible(false);
          refetch();
        }}
      />
    </>
  );
}
