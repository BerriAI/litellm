"use client";

import { useSSOSettings, type SSOSettingsValues } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Badge, Button, Card, Descriptions, Space, Typography } from "antd";
import { Shield, Trash2 } from "lucide-react";
import { useState } from "react";
import AddSSOSettingsModal from "./Modals/AddSSOSettingsModal";
import DeleteSSOSettingsModal from "./Modals/DeleteSSOSettingsModal";
import SSOSettingsEmptyPlaceholder from "./SSOSettingsEmptyPlaceholder";

const { Title, Text } = Typography;

export default function SSOSettings() {
  const { data: ssoSettings, refetch } = useSSOSettings();
  const { accessToken } = useAuthorized();
  const [isDeleteModalVisible, setIsDeleteModalVisible] = useState(false);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const isSSOConfigured =
    Boolean(ssoSettings?.values.google_client_id) ||
    Boolean(ssoSettings?.values.microsoft_client_id) ||
    Boolean(ssoSettings?.values.generic_client_id);

  // Determine the SSO provider based on the configuration
  let selectedProvider: string | null = null;
  if (ssoSettings?.values.google_client_id) {
    selectedProvider = "google";
  } else if (ssoSettings?.values.microsoft_client_id) {
    selectedProvider = "microsoft";
  } else if (ssoSettings?.values.generic_client_id) {
    // Check if it looks like Okta based on endpoints
    if (
      ssoSettings.values.generic_authorization_endpoint?.includes("okta") ||
      ssoSettings.values.generic_authorization_endpoint?.includes("auth0")
    ) {
      selectedProvider = "okta";
    } else {
      selectedProvider = "generic";
    }
  }

  const renderRedactedValue = (value?: string | null) => (
    <span className="font-mono text-gray-600">
      {value ? "••••••••••••••••••••••••••••••••" : <span className="text-gray-400 italic">Not configured</span>}
    </span>
  );

  const renderEndpointValue = (value?: string | null) => (
    <span className="font-mono text-gray-600 text-sm break-all">
      {value || <span className="text-gray-400 italic">Not configured</span>}
    </span>
  );

  const renderSimpleValue = (value?: string | null) =>
    value ? value : <span className="text-gray-400 italic">Not configured</span>;

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
      providerText: "Google OAuth",
      fields: [
        {
          label: "Client ID (Redacted)",
          render: (values: SSOSettingsValues) => renderRedactedValue(values.google_client_id),
        },
        {
          label: "Client Secret (Redacted)",
          render: (values: SSOSettingsValues) => renderRedactedValue(values.google_client_secret),
        },
        { label: "Proxy Base URL", render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url) },
      ],
    },
    microsoft: {
      providerText: "Microsoft OAuth",
      fields: [
        {
          label: "Client ID (Redacted)",
          render: (values: SSOSettingsValues) => renderRedactedValue(values.microsoft_client_id),
        },
        {
          label: "Client Secret (Redacted)",
          render: (values: SSOSettingsValues) => renderRedactedValue(values.microsoft_client_secret),
        },
        { label: "Tenant", render: (values: any) => renderSimpleValue(values.microsoft_tenant) },
        { label: "Proxy Base URL", render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url) },
      ],
    },
    okta: {
      providerText: "Okta/Auth0",
      fields: [
        {
          label: "Client ID (Redacted)",
          render: (values: SSOSettingsValues) => renderRedactedValue(values.generic_client_id),
        },
        {
          label: "Client Secret (Redacted)",
          render: (values: SSOSettingsValues) => renderRedactedValue(values.generic_client_secret),
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
      ],
    },
    generic: {
      providerText: "Generic OAuth",
      fields: [
        {
          label: "Client ID (Redacted)",
          render: (values: SSOSettingsValues) => renderRedactedValue(values.generic_client_id),
        },
        {
          label: "Client Secret (Redacted)",
          render: (values: SSOSettingsValues) => renderRedactedValue(values.generic_client_secret),
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
          <Badge status="success" text={config.providerText} />
        </Descriptions.Item>
        {config.fields.map((field, index) => (
          <Descriptions.Item key={index} label={field.label}>
            {field.render(values)}
          </Descriptions.Item>
        ))}
      </Descriptions>
    );
  };

  return (
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
              <Button danger icon={<Trash2 className="w-4 h-4" />} onClick={() => setIsDeleteModalVisible(true)}>
                Delete SSO Settings
              </Button>
            )}
          </div>
        </div>

        {isSSOConfigured ? (
          renderSSOSettings()
        ) : (
          <SSOSettingsEmptyPlaceholder onAdd={() => setIsAddModalVisible(true)} />
        )}
      </Space>

      <DeleteSSOSettingsModal
        isVisible={isDeleteModalVisible}
        onCancel={() => setIsDeleteModalVisible(false)}
        onSuccess={() => refetch()}
        accessToken={accessToken}
      />

      <AddSSOSettingsModal
        isVisible={isAddModalVisible}
        onCancel={() => setIsAddModalVisible(false)}
        onSuccess={() => {
          setIsAddModalVisible(false);
          refetch();
        }}
        accessToken={accessToken}
      />
    </Card>
  );
}
