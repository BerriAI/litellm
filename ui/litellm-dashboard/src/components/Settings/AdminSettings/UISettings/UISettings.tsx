"use client";

import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useUpdateUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUpdateUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import NotificationManager from "@/components/molecules/notifications_manager";
import PageVisibilitySettings from "./PageVisibilitySettings";
import { Alert, Card, Divider, Skeleton, Space, Switch, Typography } from "antd";
import { useTranslation } from "react-i18next";

export default function UISettings() {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const { data, isLoading, isError, error } = useUISettings();
  const { mutate: updateSettings, isPending: isUpdating, error: updateError } = useUpdateUISettings(accessToken);

  const schema = data?.field_schema;
  const property = schema?.properties?.disable_model_add_for_internal_users;
  const disableTeamAdminDeleteProperty = schema?.properties?.disable_team_admin_delete_team_user;
  const requireAuthForPublicAIHubProperty = schema?.properties?.require_auth_for_public_ai_hub;
  const forwardClientHeadersProperty = schema?.properties?.forward_client_headers_to_llm_api;
  const forwardLLMProviderAuthHeadersProperty = schema?.properties?.forward_llm_provider_auth_headers;
  const enableProjectsUIProperty = schema?.properties?.enable_projects_ui;
  const enabledPagesProperty = schema?.properties?.enabled_ui_pages_internal_users;
  const disableAgentsProperty = schema?.properties?.disable_agents_for_internal_users;
  const allowAgentsTeamAdminsProperty = schema?.properties?.allow_agents_for_team_admins;
  const disableVectorStoresProperty = schema?.properties?.disable_vector_stores_for_internal_users;
  const allowVectorStoresTeamAdminsProperty = schema?.properties?.allow_vector_stores_for_team_admins;
  const scopeUserSearchProperty = schema?.properties?.scope_user_search_to_org;
  const disableCustomApiKeysProperty = schema?.properties?.disable_custom_api_keys;
  const disableUINudgesProperty = schema?.properties?.disable_ui_nudges;
  const values = data?.values ?? {};
  const isDisabledForInternalUsers = Boolean(values.disable_model_add_for_internal_users);
  const isDisabledTeamAdminDeleteTeamUser = Boolean(values.disable_team_admin_delete_team_user);
  const isAgentsDisabled = Boolean(values.disable_agents_for_internal_users);
  const isVectorStoresDisabled = Boolean(values.disable_vector_stores_for_internal_users);

  const handleToggle = (checked: boolean) => {
    updateSettings(
      { disable_model_add_for_internal_users: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleTeamAdminDelete = (checked: boolean) => {
    updateSettings(
      { disable_team_admin_delete_team_user: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleDisableUINudges = (checked: boolean) => {
    updateSettings(
      { disable_ui_nudges: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleUpdatePageVisibility = (settings: { enabled_ui_pages_internal_users: string[] | null }) => {
    updateSettings(settings, {
      onSuccess: () => {
        NotificationManager.success(t("settingsPages.uISettings.pageVisibilityUpdateSuccess"));
      },
      onError: (error) => {
        NotificationManager.fromBackend(error);
      },
    });
  };

  const handleToggleForwardClientHeaders = (checked: boolean) => {
    updateSettings(
      { forward_client_headers_to_llm_api: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleForwardLLMProviderAuthHeaders = (checked: boolean) => {
    updateSettings(
      { forward_llm_provider_auth_headers: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleEnableProjectsUI = (checked: boolean) => {
    updateSettings(
      { enable_projects_ui: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccessRefreshing"));
          setTimeout(() => window.location.reload(), 1000);
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleRequireAuthForPublicAIHub = (checked: boolean) => {
    updateSettings(
      { require_auth_for_public_ai_hub: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleDisableAgents = (checked: boolean) => {
    updateSettings(
      { disable_agents_for_internal_users: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleAllowAgentsTeamAdmins = (checked: boolean) => {
    updateSettings(
      { allow_agents_for_team_admins: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleDisableVectorStores = (checked: boolean) => {
    updateSettings(
      { disable_vector_stores_for_internal_users: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleAllowVectorStoresTeamAdmins = (checked: boolean) => {
    updateSettings(
      { allow_vector_stores_for_team_admins: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleScopeUserSearch = (checked: boolean) => {
    updateSettings(
      { scope_user_search_to_org: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  const handleToggleDisableCustomApiKeys = (checked: boolean) => {
    updateSettings(
      { disable_custom_api_keys: checked },
      {
        onSuccess: () => {
          NotificationManager.success(t("settingsPages.uISettings.updateSuccess"));
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  return (
    <Card title={t("settingsPages.uISettings.cardTitle")}>
      {isLoading ? (
        <Skeleton active />
      ) : isError ? (
        <Alert
          type="error"
          message={t("settingsPages.uISettings.loadError")}
          description={error instanceof Error ? error.message : undefined}
        />
      ) : (
        <Space direction="vertical" size="large" style={{ width: "100%" }}>
          {schema?.description && (
            <Typography.Paragraph style={{ marginBottom: 0 }}>{schema.description}</Typography.Paragraph>
          )}

          {updateError && (
            <Alert
              type="error"
              message={t("settingsPages.uISettings.updateError")}
              description={updateError instanceof Error ? updateError.message : undefined}
            />
          )}

          <Space align="start" size="middle">
            <Switch
              checked={isDisabledForInternalUsers}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggle}
              aria-label={property?.description ?? t("settingsPages.uISettings.disableModelAddLabel")}
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>{t("settingsPages.uISettings.disableModelAddLabel")}</Typography.Text>
              {property?.description && <Typography.Text type="secondary">{property.description}</Typography.Text>}
            </Space>
          </Space>

          <Space align="start" size="middle">
            <Switch
              checked={isDisabledTeamAdminDeleteTeamUser}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleTeamAdminDelete}
              aria-label={
                disableTeamAdminDeleteProperty?.description ?? t("settingsPages.uISettings.disableTeamAdminDeleteLabel")
              }
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>{t("settingsPages.uISettings.disableTeamAdminDeleteLabel")}</Typography.Text>
              {disableTeamAdminDeleteProperty?.description && (
                <Typography.Text type="secondary">{disableTeamAdminDeleteProperty.description}</Typography.Text>
              )}
            </Space>
          </Space>

          <Space align="start" size="middle">
            <Switch
              checked={values.require_auth_for_public_ai_hub}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleRequireAuthForPublicAIHub}
              aria-label={
                requireAuthForPublicAIHubProperty?.description ??
                t("settingsPages.uISettings.requireAuthPublicAIHubLabel")
              }
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>{t("settingsPages.uISettings.requireAuthPublicAIHubLabel")}</Typography.Text>
              {requireAuthForPublicAIHubProperty?.description && (
                <Typography.Text type="secondary">{requireAuthForPublicAIHubProperty.description}</Typography.Text>
              )}
            </Space>
          </Space>

          <Space align="start" size="middle">
            <Switch
              checked={Boolean(values.forward_client_headers_to_llm_api)}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleForwardClientHeaders}
              aria-label={
                forwardClientHeadersProperty?.description ?? t("settingsPages.uISettings.forwardClientHeadersLabel")
              }
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>{t("settingsPages.uISettings.forwardClientHeadersLabel")}</Typography.Text>
              <Typography.Text type="secondary">
                {forwardClientHeadersProperty?.description ?? t("settingsPages.uISettings.forwardClientHeadersDesc")}
              </Typography.Text>
            </Space>
          </Space>

          <Space align="start" size="middle">
            <Switch
              checked={Boolean(values.forward_llm_provider_auth_headers)}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleForwardLLMProviderAuthHeaders}
              aria-label={
                forwardLLMProviderAuthHeadersProperty?.description ??
                t("settingsPages.uISettings.forwardLLMProviderAuthHeadersLabel")
              }
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>
                {t("settingsPages.uISettings.forwardLLMProviderAuthHeadersLabel")}
              </Typography.Text>
              <Typography.Text type="secondary">
                {forwardLLMProviderAuthHeadersProperty?.description ??
                  t("settingsPages.uISettings.forwardLLMProviderAuthHeadersDesc")}
              </Typography.Text>
            </Space>
          </Space>

          {enableProjectsUIProperty && (
            <Space align="start" size="middle">
              <Switch
                checked={Boolean(values.enable_projects_ui)}
                disabled={isUpdating}
                loading={isUpdating}
                onChange={handleToggleEnableProjectsUI}
                aria-label={enableProjectsUIProperty.description ?? t("settingsPages.uISettings.enableProjectsUILabel")}
              />
              <Space direction="vertical" size={4}>
                <Typography.Text strong>{t("settingsPages.uISettings.enableProjectsUILabel")}</Typography.Text>
                <Typography.Text type="secondary">
                  {enableProjectsUIProperty.description ?? t("settingsPages.uISettings.enableProjectsUIDesc")}
                </Typography.Text>
              </Space>
            </Space>
          )}

          <Divider />

          {/* Agents access control */}
          <Space align="start" size="middle">
            <Switch
              checked={isAgentsDisabled}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleDisableAgents}
              aria-label={disableAgentsProperty?.description ?? t("settingsPages.uISettings.disableAgentsLabel")}
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>{t("settingsPages.uISettings.disableAgentsLabel")}</Typography.Text>
              {disableAgentsProperty?.description && (
                <Typography.Text type="secondary">{disableAgentsProperty.description}</Typography.Text>
              )}
            </Space>
          </Space>

          <Space align="start" size="middle" style={{ marginLeft: 32 }}>
            <Switch
              checked={Boolean(values.allow_agents_for_team_admins)}
              disabled={isUpdating || !isAgentsDisabled}
              loading={isUpdating}
              onChange={handleToggleAllowAgentsTeamAdmins}
              aria-label={
                allowAgentsTeamAdminsProperty?.description ?? t("settingsPages.uISettings.allowAgentsTeamAdminsLabel")
              }
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong type={!isAgentsDisabled ? "secondary" : undefined}>
                {t("settingsPages.uISettings.allowAgentsTeamAdminsLabel")}
              </Typography.Text>
              {allowAgentsTeamAdminsProperty?.description && (
                <Typography.Text type="secondary">{allowAgentsTeamAdminsProperty.description}</Typography.Text>
              )}
            </Space>
          </Space>

          <Divider />

          {/* Vector Stores access control */}
          <Space align="start" size="middle">
            <Switch
              checked={isVectorStoresDisabled}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleDisableVectorStores}
              aria-label={
                disableVectorStoresProperty?.description ?? t("settingsPages.uISettings.disableVectorStoresLabel")
              }
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>{t("settingsPages.uISettings.disableVectorStoresLabel")}</Typography.Text>
              {disableVectorStoresProperty?.description && (
                <Typography.Text type="secondary">{disableVectorStoresProperty.description}</Typography.Text>
              )}
            </Space>
          </Space>

          <Space align="start" size="middle" style={{ marginLeft: 32 }}>
            <Switch
              checked={Boolean(values.allow_vector_stores_for_team_admins)}
              disabled={isUpdating || !isVectorStoresDisabled}
              loading={isUpdating}
              onChange={handleToggleAllowVectorStoresTeamAdmins}
              aria-label={
                allowVectorStoresTeamAdminsProperty?.description ??
                t("settingsPages.uISettings.allowVectorStoresTeamAdminsLabel")
              }
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong type={!isVectorStoresDisabled ? "secondary" : undefined}>
                {t("settingsPages.uISettings.allowVectorStoresTeamAdminsLabel")}
              </Typography.Text>
              {allowVectorStoresTeamAdminsProperty?.description && (
                <Typography.Text type="secondary">{allowVectorStoresTeamAdminsProperty.description}</Typography.Text>
              )}
            </Space>
          </Space>

          <Divider />

          {/* Scope user search to organization */}
          <Space align="start" size="middle">
            <Switch
              checked={Boolean(values.scope_user_search_to_org)}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleScopeUserSearch}
              aria-label={scopeUserSearchProperty?.description ?? t("settingsPages.uISettings.scopeUserSearchLabel")}
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>{t("settingsPages.uISettings.scopeUserSearchLabel")}</Typography.Text>
              <Typography.Text type="secondary">
                {scopeUserSearchProperty?.description ?? t("settingsPages.uISettings.scopeUserSearchDesc")}
              </Typography.Text>
            </Space>
          </Space>

          <Divider />

          {/* Disable custom Virtual key values */}
          <Space align="start" size="middle">
            <Switch
              checked={Boolean(values.disable_custom_api_keys)}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleDisableCustomApiKeys}
              aria-label={
                disableCustomApiKeysProperty?.description ?? t("settingsPages.uISettings.disableCustomApiKeysLabel")
              }
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>{t("settingsPages.uISettings.disableCustomApiKeysLabel")}</Typography.Text>
              <Typography.Text type="secondary">
                {disableCustomApiKeysProperty?.description ?? t("settingsPages.uISettings.disableCustomApiKeysDesc")}
              </Typography.Text>
            </Space>
          </Space>

          <Divider />

          {/* Disable in-product UI nudges */}
          <Space align="start" size="middle">
            <Switch
              checked={Boolean(values.disable_ui_nudges)}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleDisableUINudges}
              aria-label={disableUINudgesProperty?.description ?? t("settingsPages.uISettings.disableUINudgesLabel")}
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>{t("settingsPages.uISettings.disableUINudgesLabel")}</Typography.Text>
              <Typography.Text type="secondary">
                {disableUINudgesProperty?.description ?? t("settingsPages.uISettings.disableUINudgesDesc")}
              </Typography.Text>
            </Space>
          </Space>

          <Divider />

          {/* Page Visibility for Internal Users */}
          <PageVisibilitySettings
            enabledPagesInternalUsers={values.enabled_ui_pages_internal_users}
            enabledPagesPropertyDescription={enabledPagesProperty?.description}
            isUpdating={isUpdating}
            onUpdate={handleUpdatePageVisibility}
          />
        </Space>
      )}
    </Card>
  );
}
