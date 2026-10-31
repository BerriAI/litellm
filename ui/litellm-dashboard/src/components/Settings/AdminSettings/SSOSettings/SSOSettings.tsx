"use client";

import { Copy, Edit, Shield, Trash2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { useSSOSettings, type SSOSettingsValues } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { Logo } from "@/components/molecules/logo/Logo";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { copyToClipboard } from "@/utils/dataUtils";

import AddSSOSettingsModal from "./Modals/AddSSOSettingsModal";
import DeleteSSOSettingsModal from "./Modals/DeleteSSOSettingsModal";
import EditSSOSettingsModal from "./Modals/EditSSOSettingsModal";
import RedactableField from "./RedactableField";
import RoleMappings from "./RoleMappings";
import SSOSettingsEmptyPlaceholder from "./SSOSettingsEmptyPlaceholder";
import SSOSettingsLoadingSkeleton from "./SSOSettingsLoadingSkeleton";
import { ssoProviderDisplayNames, ssoProviderLogoMap } from "./constants";
import { detectSSOProvider } from "./utils";

function NotConfigured() {
  const { t } = useTranslation();
  return <span className="text-muted-foreground italic">{t("adminPanel.ssoSettings.details.notConfigured")}</span>;
}

function DetailRow({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3">
      <dt className="bg-muted/50 px-4 py-3 text-sm font-medium text-foreground">{label}</dt>
      <dd className="min-w-0 px-4 py-3 text-sm text-foreground sm:col-span-2">{children}</dd>
    </div>
  );
}

function EndpointValue({ value }: { value?: string | null }) {
  const { t } = useTranslation();
  if (!value) return <span className="font-mono text-muted-foreground">-</span>;

  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="truncate font-mono text-sm text-muted-foreground">{value}</span>
      <Button
        type="button"
        variant="ghost"
        size="icon-sm"
        aria-label={t("adminPanel.ssoSettings.details.copyValue")}
        onClick={() => void copyToClipboard(value, t("adminPanel.ssoSettings.details.copied"))}
      >
        <Copy className="size-3.5" />
      </Button>
    </div>
  );
}

export default function SSOSettings() {
  const { t } = useTranslation();
  const { data: ssoSettings, refetch, isLoading } = useSSOSettings();
  const [isDeleteModalVisible, setIsDeleteModalVisible] = useState(false);
  const [isAddModalVisible, setIsAddModalVisible] = useState(false);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const isSSOConfigured = [
    ssoSettings?.values.google_client_id,
    ssoSettings?.values.microsoft_client_id,
    ssoSettings?.values.generic_client_id,
    ssoSettings?.values.saml_idp_metadata_url,
    ssoSettings?.values.saml_idp_metadata_xml,
  ].some(Boolean);
  const selectedProvider = ssoSettings?.values ? detectSSOProvider(ssoSettings.values) : null;
  const isRoleMappingsEnabled = Boolean(ssoSettings?.values.role_mappings);
  const isTeamMappingsEnabled = Boolean(ssoSettings?.values.team_mappings);

  const renderSimpleValue = (value?: string | null) => value || <NotConfigured />;
  const renderTeamMappingsField = (values: SSOSettingsValues) =>
    values.team_mappings?.team_ids_jwt_field ? (
      <Badge variant="secondary">{values.team_mappings.team_ids_jwt_field}</Badge>
    ) : (
      <NotConfigured />
    );

  const providerConfigs = {
    google: {
      providerText: t(ssoProviderDisplayNames.google),
      fields: [
        {
          label: t("adminPanel.ssoSettings.details.clientId"),
          render: (values: SSOSettingsValues) => <RedactableField value={values.google_client_id} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.clientSecret"),
          render: (values: SSOSettingsValues) => <RedactableField value={values.google_client_secret} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.proxyBaseUrl"),
          render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url),
        },
      ],
    },
    microsoft: {
      providerText: t(ssoProviderDisplayNames.microsoft),
      fields: [
        {
          label: t("adminPanel.ssoSettings.details.clientId"),
          render: (values: SSOSettingsValues) => <RedactableField value={values.microsoft_client_id} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.clientSecret"),
          render: (values: SSOSettingsValues) => <RedactableField value={values.microsoft_client_secret} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.tenant"),
          render: (values: SSOSettingsValues) => renderSimpleValue(values.microsoft_tenant),
        },
        {
          label: t("adminPanel.ssoSettings.details.proxyBaseUrl"),
          render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url),
        },
      ],
    },
    okta: {
      providerText: t(ssoProviderDisplayNames.okta),
      fields: [
        {
          label: t("adminPanel.ssoSettings.details.clientId"),
          render: (values: SSOSettingsValues) => <RedactableField value={values.generic_client_id} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.clientSecret"),
          render: (values: SSOSettingsValues) => <RedactableField value={values.generic_client_secret} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.authorizationEndpoint"),
          render: (values: SSOSettingsValues) => <EndpointValue value={values.generic_authorization_endpoint} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.tokenEndpoint"),
          render: (values: SSOSettingsValues) => <EndpointValue value={values.generic_token_endpoint} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.userInfoEndpoint"),
          render: (values: SSOSettingsValues) => <EndpointValue value={values.generic_userinfo_endpoint} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.scopes"),
          render: (values: SSOSettingsValues) => renderSimpleValue(values.generic_scope),
        },
        {
          label: t("adminPanel.ssoSettings.details.proxyBaseUrl"),
          render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url),
        },
        isTeamMappingsEnabled
          ? {
              label: t("adminPanel.ssoSettings.details.teamIdsJwtField"),
              render: (values: SSOSettingsValues) => renderTeamMappingsField(values),
            }
          : null,
      ],
    },
    generic: {
      providerText: t(ssoProviderDisplayNames.generic),
      fields: [
        {
          label: t("adminPanel.ssoSettings.details.clientId"),
          render: (values: SSOSettingsValues) => <RedactableField value={values.generic_client_id} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.clientSecret"),
          render: (values: SSOSettingsValues) => <RedactableField value={values.generic_client_secret} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.authorizationEndpoint"),
          render: (values: SSOSettingsValues) => <EndpointValue value={values.generic_authorization_endpoint} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.tokenEndpoint"),
          render: (values: SSOSettingsValues) => <EndpointValue value={values.generic_token_endpoint} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.userInfoEndpoint"),
          render: (values: SSOSettingsValues) => <EndpointValue value={values.generic_userinfo_endpoint} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.scopes"),
          render: (values: SSOSettingsValues) => renderSimpleValue(values.generic_scope),
        },
        {
          label: t("adminPanel.ssoSettings.details.proxyBaseUrl"),
          render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url),
        },
        isTeamMappingsEnabled
          ? {
              label: t("adminPanel.ssoSettings.details.teamIdsJwtField"),
              render: (values: SSOSettingsValues) => renderTeamMappingsField(values),
            }
          : null,
      ],
    },
    saml: {
      providerText: t(ssoProviderDisplayNames.saml),
      fields: [
        {
          label: t("adminPanel.ssoSettings.details.samlIdpMetadataUrl"),
          render: (values: SSOSettingsValues) => <EndpointValue value={values.saml_idp_metadata_url} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.samlIdpMetadataXml"),
          render: (values: SSOSettingsValues) =>
            values.saml_idp_metadata_xml ? (
              <Badge variant="secondary">{t("adminPanel.ssoSettings.details.provided")}</Badge>
            ) : (
              <NotConfigured />
            ),
        },
        {
          label: t("adminPanel.ssoSettings.details.samlSpEntityId"),
          render: (values: SSOSettingsValues) => <EndpointValue value={values.saml_sp_entity_id} />,
        },
        {
          label: t("adminPanel.ssoSettings.details.samlAllowUnsolicited"),
          render: (values: SSOSettingsValues) => (
            <Badge variant={values.saml_allow_unsolicited === "true" ? "default" : "secondary"}>
              {values.saml_allow_unsolicited === "true"
                ? t("adminPanel.ssoSettings.details.enabled")
                : t("adminPanel.ssoSettings.details.disabled")}
            </Badge>
          ),
        },
        {
          label: t("adminPanel.ssoSettings.details.proxyBaseUrl"),
          render: (values: SSOSettingsValues) => renderSimpleValue(values.proxy_base_url),
        },
      ],
    },
  };

  const renderSSOSettings = () => {
    if (!ssoSettings?.values || !selectedProvider) return null;
    const config = providerConfigs[selectedProvider as keyof typeof providerConfigs];
    if (!config) return null;

    return (
      <dl className="divide-y divide-border overflow-hidden rounded-md border border-border">
        <DetailRow label={t("adminPanel.ssoSettings.details.provider")}>
          <div className="flex items-center gap-2">
            {ssoProviderLogoMap[selectedProvider] && (
              <Logo
                src={ssoProviderLogoMap[selectedProvider]}
                label={
                  ssoProviderDisplayNames[selectedProvider]
                    ? t(ssoProviderDisplayNames[selectedProvider])
                    : selectedProvider
                }
                className="size-6 object-contain"
              />
            )}
            <span>{config.providerText}</span>
          </div>
        </DetailRow>
        {config.fields.map(
          (field) =>
            field && (
              <DetailRow key={field.label} label={field.label}>
                {field.render(ssoSettings.values)}
              </DetailRow>
            ),
        )}
      </dl>
    );
  };

  return (
    <>
      {isLoading ? (
        <SSOSettingsLoadingSkeleton />
      ) : (
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <div className="flex items-center gap-3">
                <Shield className="size-6 text-muted-foreground" />
                <div>
                  <CardTitle>
                    <h3>{t("adminPanel.ssoSettings.header.title")}</h3>
                  </CardTitle>
                  <CardDescription>{t("adminPanel.ssoSettings.header.description")}</CardDescription>
                </div>
              </div>
              {isSSOConfigured && (
                <CardAction className="flex gap-2">
                  <Button type="button" variant="outline" onClick={() => setIsEditModalVisible(true)}>
                    <Edit />
                    {t("adminPanel.ssoSettings.actions.edit")}
                  </Button>
                  <Button type="button" variant="destructive" onClick={() => setIsDeleteModalVisible(true)}>
                    <Trash2 />
                    {t("adminPanel.ssoSettings.actions.delete")}
                  </Button>
                </CardAction>
              )}
            </CardHeader>
            <CardContent>
              {isSSOConfigured ? (
                renderSSOSettings()
              ) : (
                <SSOSettingsEmptyPlaceholder onAdd={() => setIsAddModalVisible(true)} />
              )}
            </CardContent>
          </Card>
          {isRoleMappingsEnabled && <RoleMappings roleMappings={ssoSettings?.values.role_mappings} />}
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
