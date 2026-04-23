"use client";

import {
  useSSOSettings,
  type SSOSettingsValues,
} from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { Check, Copy, Edit, Shield, Trash2 } from "lucide-react";
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

const CopyableEndpoint: React.FC<{ value?: string | null }> = ({ value }) => {
  const [copied, setCopied] = useState(false);
  if (!value) {
    return (
      <span className="font-mono text-muted-foreground text-sm">-</span>
    );
  }
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_e) {
      /* noop */
    }
  };
  return (
    <span className="inline-flex items-center gap-2 font-mono text-muted-foreground text-sm">
      <span>{value}</span>
      <button
        type="button"
        onClick={handleCopy}
        className="text-muted-foreground hover:text-foreground"
        aria-label="Copy endpoint"
      >
        {copied ? (
          <Check className="h-3 w-3 text-emerald-500" />
        ) : (
          <Copy className="h-3 w-3" />
        )}
      </button>
    </span>
  );
};

const DescRow: React.FC<{
  label: string;
  children: React.ReactNode;
  first?: boolean;
}> = ({ label, children, first }) => (
  <div
    className={cn(
      "grid grid-cols-[minmax(180px,240px)_1fr]",
      !first && "border-t border-border",
    )}
  >
    <div className="bg-muted px-4 py-3 font-medium text-sm">{label}</div>
    <div className="px-4 py-3 text-sm">{children}</div>
  </div>
);

export default function SSOSettings() {
  const { data: ssoSettings, refetch, isLoading } = useSSOSettings();
  const [isDeleteModalVisible, setIsDeleteModalVisible] = useState(false);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const isSSOConfigured =
    Boolean(ssoSettings?.values.google_client_id) ||
    Boolean(ssoSettings?.values.microsoft_client_id) ||
    Boolean(ssoSettings?.values.generic_client_id);

  const selectedProvider = ssoSettings?.values
    ? detectSSOProvider(ssoSettings.values)
    : null;
  const isTeamMappingsEnabled = Boolean(ssoSettings?.values.team_mappings);
  const isRoleMappingsEnabled = Boolean(ssoSettings?.values.role_mappings);

  const renderSimpleValue = (value?: string | null) =>
    value ? (
      <span>{value}</span>
    ) : (
      <span className="text-muted-foreground italic">Not configured</span>
    );

  const renderTeamMappingsField = (values: SSOSettingsValues) => {
    if (!values.team_mappings?.team_ids_jwt_field) {
      return (
        <span className="text-muted-foreground italic">Not configured</span>
      );
    }
    return (
      <Badge variant="secondary">{values.team_mappings.team_ids_jwt_field}</Badge>
    );
  };

  const providerConfigs = {
    google: {
      providerText: ssoProviderDisplayNames.google,
      fields: [
        {
          label: "Client ID",
          render: (values: SSOSettingsValues) => (
            <RedactableField value={values.google_client_id} />
          ),
        },
        {
          label: "Client Secret",
          render: (values: SSOSettingsValues) => (
            <RedactableField value={values.google_client_secret} />
          ),
        },
        {
          label: "Proxy Base URL",
          render: (values: SSOSettingsValues) =>
            renderSimpleValue(values.proxy_base_url),
        },
      ],
    },
    microsoft: {
      providerText: ssoProviderDisplayNames.microsoft,
      fields: [
        {
          label: "Client ID",
          render: (values: SSOSettingsValues) => (
            <RedactableField value={values.microsoft_client_id} />
          ),
        },
        {
          label: "Client Secret",
          render: (values: SSOSettingsValues) => (
            <RedactableField value={values.microsoft_client_secret} />
          ),
        },
        {
          label: "Tenant",
          render: (values: SSOSettingsValues) =>
            renderSimpleValue(values.microsoft_tenant),
        },
        {
          label: "Proxy Base URL",
          render: (values: SSOSettingsValues) =>
            renderSimpleValue(values.proxy_base_url),
        },
      ],
    },
    okta: {
      providerText: ssoProviderDisplayNames.okta,
      fields: [
        {
          label: "Client ID",
          render: (values: SSOSettingsValues) => (
            <RedactableField value={values.generic_client_id} />
          ),
        },
        {
          label: "Client Secret",
          render: (values: SSOSettingsValues) => (
            <RedactableField value={values.generic_client_secret} />
          ),
        },
        {
          label: "Authorization Endpoint",
          render: (values: SSOSettingsValues) => (
            <CopyableEndpoint value={values.generic_authorization_endpoint} />
          ),
        },
        {
          label: "Token Endpoint",
          render: (values: SSOSettingsValues) => (
            <CopyableEndpoint value={values.generic_token_endpoint} />
          ),
        },
        {
          label: "User Info Endpoint",
          render: (values: SSOSettingsValues) => (
            <CopyableEndpoint value={values.generic_userinfo_endpoint} />
          ),
        },
        {
          label: "Proxy Base URL",
          render: (values: SSOSettingsValues) =>
            renderSimpleValue(values.proxy_base_url),
        },
        isTeamMappingsEnabled
          ? {
              label: "Team IDs JWT Field",
              render: (values: SSOSettingsValues) =>
                renderTeamMappingsField(values),
            }
          : null,
      ],
    },
    generic: {
      providerText: ssoProviderDisplayNames.generic,
      fields: [
        {
          label: "Client ID",
          render: (values: SSOSettingsValues) => (
            <RedactableField value={values.generic_client_id} />
          ),
        },
        {
          label: "Client Secret",
          render: (values: SSOSettingsValues) => (
            <RedactableField value={values.generic_client_secret} />
          ),
        },
        {
          label: "Authorization Endpoint",
          render: (values: SSOSettingsValues) => (
            <CopyableEndpoint value={values.generic_authorization_endpoint} />
          ),
        },
        {
          label: "Token Endpoint",
          render: (values: SSOSettingsValues) => (
            <CopyableEndpoint value={values.generic_token_endpoint} />
          ),
        },
        {
          label: "User Info Endpoint",
          render: (values: SSOSettingsValues) => (
            <CopyableEndpoint value={values.generic_userinfo_endpoint} />
          ),
        },
        {
          label: "Proxy Base URL",
          render: (values: SSOSettingsValues) =>
            renderSimpleValue(values.proxy_base_url),
        },
        isTeamMappingsEnabled
          ? {
              label: "Team IDs JWT Field",
              render: (values: SSOSettingsValues) =>
                renderTeamMappingsField(values),
            }
          : null,
      ],
    },
  };

  const renderSSOSettings = () => {
    if (!ssoSettings?.values || !selectedProvider) return null;

    const { values } = ssoSettings;
    const config =
      providerConfigs[selectedProvider as keyof typeof providerConfigs];

    if (!config) return null;

    return (
      <div className="border border-border rounded-md overflow-hidden">
        <DescRow label="Provider" first>
          <div className="flex items-center gap-2">
            {ssoProviderLogoMap[selectedProvider] && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={ssoProviderLogoMap[selectedProvider]}
                alt={selectedProvider}
                className="h-6 w-6 object-contain"
              />
            )}
            <span>{config.providerText}</span>
          </div>
        </DescRow>
        {config.fields.map(
          (field, index) =>
            field && (
              <DescRow key={index} label={field.label}>
                {field.render(values)}
              </DescRow>
            ),
        )}
      </div>
    );
  };

  return (
    <>
      {isLoading ? (
        <SSOSettingsLoadingSkeleton />
      ) : (
        <div className="flex flex-col gap-6 w-full">
          <Card className="p-6">
            <div className="flex flex-col gap-6 w-full">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Shield className="w-6 h-6 text-muted-foreground" />
                  <div>
                    <h3 className="text-xl font-semibold">
                      SSO Configuration
                    </h3>
                    <p className="text-muted-foreground">
                      Manage Single Sign-On authentication settings
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  {isSSOConfigured && (
                    <>
                      <Button
                        variant="outline"
                        onClick={() => setIsEditModalVisible(true)}
                      >
                        <Edit className="w-4 h-4" />
                        Edit SSO Settings
                      </Button>
                      <Button
                        variant="destructive"
                        onClick={() => setIsDeleteModalVisible(true)}
                      >
                        <Trash2 className="w-4 h-4" />
                        Delete SSO Settings
                      </Button>
                    </>
                  )}
                </div>
              </div>

              {isSSOConfigured ? (
                renderSSOSettings()
              ) : (
                <SSOSettingsEmptyPlaceholder
                  onAdd={() => setIsAddModalVisible(true)}
                />
              )}
            </div>
          </Card>
          {isRoleMappingsEnabled && (
            <RoleMappings roleMappings={ssoSettings?.values.role_mappings} />
          )}
        </div>
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
