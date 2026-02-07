"use client";

import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useUpdateUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUpdateUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import NotificationManager from "@/components/molecules/notifications_manager";
import PageVisibilitySettings from "./PageVisibilitySettings";
import { Alert, Card, Divider, Skeleton, Space, Switch, Typography } from "antd";

export default function UISettings() {
  const { accessToken } = useAuthorized();
  const { data, isLoading, isError, error } = useUISettings();
  const { mutate: updateSettings, isPending: isUpdating, error: updateError } = useUpdateUISettings(accessToken);

  const schema = data?.field_schema;
  const property = schema?.properties?.disable_model_add_for_internal_users;
  const disableTeamAdminDeleteProperty = schema?.properties?.disable_team_admin_delete_team_user;
  const requireAuthForPublicAIHubProperty = schema?.properties?.require_auth_for_public_ai_hub;
  const enabledPagesProperty = schema?.properties?.enabled_ui_pages_internal_users;
  const values = data?.values ?? {};
  const isDisabledForInternalUsers = Boolean(values.disable_model_add_for_internal_users);
  const isDisabledTeamAdminDeleteTeamUser = Boolean(values.disable_team_admin_delete_team_user);

  const handleToggle = (checked: boolean) => {
    updateSettings(
      { disable_model_add_for_internal_users: checked },
      {
        onSuccess: () => {
          NotificationManager.success("UI settings updated successfully");
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
          NotificationManager.success("UI settings updated successfully");
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
        NotificationManager.success("Page visibility settings updated successfully");
      },
      onError: (error) => {
        NotificationManager.fromBackend(error);
      },
    });
  };

  const handleToggleRequireAuthForPublicAIHub = (checked: boolean) => {
    updateSettings(
      { require_auth_for_public_ai_hub: checked },
      {
        onSuccess: () => {
          NotificationManager.success("UI settings updated successfully");
        },
        onError: (error) => {
          NotificationManager.fromBackend(error);
        },
      },
    );
  };

  return (
    <Card title="UI Settings">
      {isLoading ? (
        <Skeleton active />
      ) : isError ? (
        <Alert
          type="error"
          message="Could not load UI settings"
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
              message="Could not update UI settings"
              description={updateError instanceof Error ? updateError.message : undefined}
            />
          )}

          <Space align="start" size="middle">
            <Switch
              checked={isDisabledForInternalUsers}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggle}
              aria-label={property?.description ?? "Disable model add for internal users"}
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>Disable model add for internal users</Typography.Text>
              {property?.description && <Typography.Text type="secondary">{property.description}</Typography.Text>}
            </Space>
          </Space>

          <Space align="start" size="middle">
            <Switch
              checked={isDisabledTeamAdminDeleteTeamUser}
              disabled={isUpdating}
              loading={isUpdating}
              onChange={handleToggleTeamAdminDelete}
              aria-label={disableTeamAdminDeleteProperty?.description ?? "Disable team admin delete team user"}
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>Disable team admin delete team user</Typography.Text>
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
              aria-label={requireAuthForPublicAIHubProperty?.description ?? "Require authentication for public AI Hub"}
            />
            <Space direction="vertical" size={4}>
              <Typography.Text strong>Require authentication for public AI Hub</Typography.Text>
              {requireAuthForPublicAIHubProperty?.description && (
                <Typography.Text type="secondary">{requireAuthForPublicAIHubProperty.description}</Typography.Text>
              )}
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
