"use client";

import { useTranslation } from "react-i18next";
import { useTerminology } from "@/contexts/TerminologyContext";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useUpdateUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUpdateUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { toast } from "@/lib/toast";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import { uiHref } from "@/utils/uiHref";
import { relativeTimeOptions } from "@/components/shared/date_range_presets";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import PageVisibilitySettings from "./PageVisibilitySettings";
import BackgroundChatSettings from "./BackgroundChatSettings";

interface SettingRowProps {
  ariaLabel: string;
  checked: boolean;
  description?: string;
  disabled: boolean;
  indented?: boolean;
  label: string;
  muted?: boolean;
  onCheckedChange: (checked: boolean) => void;
}

function SettingRow({
  ariaLabel,
  checked,
  description,
  disabled,
  indented = false,
  label,
  muted = false,
  onCheckedChange,
}: SettingRowProps) {
  return (
    <div className={indented ? "ml-8 flex items-start gap-3" : "flex items-start gap-3"}>
      <Switch checked={checked} disabled={disabled} onCheckedChange={onCheckedChange} aria-label={ariaLabel} />
      <div className="space-y-1">
        <p className={muted ? "text-sm font-medium text-muted-foreground" : "text-sm font-medium text-foreground"}>
          {label}
        </p>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>
    </div>
  );
}

export default function UISettings() {
  const { t } = useTranslation();
  const term = useTerminology();
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
  const enableChatUIProperty = schema?.properties?.enable_chat_ui;
  const enabledPagesProperty = schema?.properties?.enabled_ui_pages_internal_users;
  const disableAgentsProperty = schema?.properties?.disable_agents_for_internal_users;
  const allowAgentsTeamAdminsProperty = schema?.properties?.allow_agents_for_team_admins;
  const disableVectorStoresProperty = schema?.properties?.disable_vector_stores_for_internal_users;
  const allowVectorStoresTeamAdminsProperty = schema?.properties?.allow_vector_stores_for_team_admins;
  const scopeUserSearchProperty = schema?.properties?.scope_user_search_to_org;
  const disableCustomApiKeysProperty = schema?.properties?.disable_custom_api_keys;
  const values = data?.values ?? {};
  const usageRange =
    relativeTimeOptions.find((option) => option.settingValue === values.usage_default_time_range)?.settingValue ??
    "last_7_days";
  const usageRangeConfigOwned = data?.source?.usage_default_time_range === "config";
  const moyaiUrl = (values.moyai_url as string | undefined) ?? null;
  const isDisabledForInternalUsers = Boolean(values.disable_model_add_for_internal_users);
  const isDisabledTeamAdminDeleteTeamUser = Boolean(values.disable_team_admin_delete_team_user);
  const isAgentsDisabled = Boolean(values.disable_agents_for_internal_users);
  const isVectorStoresDisabled = Boolean(values.disable_vector_stores_for_internal_users);

  const handleToggle = (checked: boolean) => {
    updateSettings(
      { disable_model_add_for_internal_users: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleTeamAdminDelete = (checked: boolean) => {
    updateSettings(
      { disable_team_admin_delete_team_user: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleUpdatePageVisibility = (settings: { enabled_ui_pages_internal_users: string[] | null }) => {
    updateSettings(settings, {
      onSuccess: () => {
        toast.success(t("adminPanel.uiSettings.toasts.pageVisibilityUpdated"));
      },
      onError: (error) => {
        toast.fromError(error);
      },
    });
  };

  const handleToggleForwardClientHeaders = (checked: boolean) => {
    updateSettings(
      { forward_client_headers_to_llm_api: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleForwardLLMProviderAuthHeaders = (checked: boolean) => {
    updateSettings(
      { forward_llm_provider_auth_headers: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleEnableProjectsUI = (checked: boolean) => {
    updateSettings(
      { enable_projects_ui: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updatedRefreshing"));
          setTimeout(() => window.location.reload(), 1000);
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleEnableChatUI = (checked: boolean) => {
    updateSettings(
      { enable_chat_ui: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updatedRefreshing"));
          setTimeout(() => window.location.reload(), 1000);
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleRequireAuthForPublicAIHub = (checked: boolean) => {
    updateSettings(
      { require_auth_for_public_ai_hub: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleDisableAgents = (checked: boolean) => {
    updateSettings(
      { disable_agents_for_internal_users: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleAllowAgentsTeamAdmins = (checked: boolean) => {
    updateSettings(
      { allow_agents_for_team_admins: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleDisableVectorStores = (checked: boolean) => {
    updateSettings(
      { disable_vector_stores_for_internal_users: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleAllowVectorStoresTeamAdmins = (checked: boolean) => {
    updateSettings(
      { allow_vector_stores_for_team_admins: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleScopeUserSearch = (checked: boolean) => {
    updateSettings(
      { scope_user_search_to_org: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleDisconnectMoyai = () => {
    updateSettings(
      { moyai_url: null },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  const handleToggleDisableCustomApiKeys = (checked: boolean) => {
    updateSettings(
      { disable_custom_api_keys: checked },
      {
        onSuccess: () => {
          toast.success(t("adminPanel.uiSettings.toasts.updated"));
        },
        onError: (error) => {
          toast.fromError(error);
        },
      },
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          <h3>{t("adminPanel.uiSettings.title")}</h3>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div role="status" aria-label={t("adminPanel.uiSettings.loadingAriaLabel")} className="space-y-3">
            <Skeleton className="h-5 w-72" />
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : isError ? (
          <Alert variant="error">
            <AlertTitle>{t("adminPanel.uiSettings.errors.loadFailed")}</AlertTitle>
            {error instanceof Error && <AlertDescription>{error.message}</AlertDescription>}
          </Alert>
        ) : (
          <div className="space-y-6">
            {schema?.description && <p className="text-sm text-foreground">{schema.description}</p>}
            {updateError && (
              <Alert variant="error">
                <AlertTitle>{t("adminPanel.uiSettings.errors.updateFailed")}</AlertTitle>
                {updateError instanceof Error && <AlertDescription>{updateError.message}</AlertDescription>}
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="usage-default-time-range">{t("adminPanel.uiSettings.usageRange.label")}</Label>
              <p id="usage-default-time-range-help" className="text-sm text-muted-foreground">
                {t("adminPanel.uiSettings.usageRange.help")}
              </p>
              <Select
                value={usageRange}
                disabled={isUpdating || usageRangeConfigOwned}
                items={relativeTimeOptions.map((option) => ({ value: option.settingValue, label: option.label }))}
                onValueChange={(value) => {
                  if (value === null) return;
                  updateSettings(
                    { usage_default_time_range: value },
                    {
                      onSuccess: () => toast.success(t("adminPanel.uiSettings.toasts.updated")),
                      onError: (error) => toast.fromError(error),
                    },
                  );
                }}
              >
                <SelectTrigger
                  id="usage-default-time-range"
                  aria-describedby="usage-default-time-range-help"
                  className="w-64"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {relativeTimeOptions.map((option) => (
                    <SelectItem key={option.settingValue} value={option.settingValue}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {usageRangeConfigOwned && (
                <p className="text-sm text-muted-foreground">{t("adminPanel.uiSettings.usageRange.managedByConfig")}</p>
              )}
            </div>
            <SettingRow
              checked={isDisabledForInternalUsers}
              disabled={isUpdating}
              onCheckedChange={handleToggle}
              ariaLabel={property?.description ?? t("adminPanel.uiSettings.settings.disableModelAdd.label")}
              label={t("adminPanel.uiSettings.settings.disableModelAdd.label")}
              description={property?.description}
            />
            <SettingRow
              checked={isDisabledTeamAdminDeleteTeamUser}
              disabled={isUpdating}
              onCheckedChange={handleToggleTeamAdminDelete}
              ariaLabel={
                disableTeamAdminDeleteProperty?.description ??
                t("adminPanel.uiSettings.settings.disableTeamAdminDelete.label")
              }
              label={t("adminPanel.uiSettings.settings.disableTeamAdminDelete.label")}
              description={disableTeamAdminDeleteProperty?.description}
            />
            <SettingRow
              checked={Boolean(values.require_auth_for_public_ai_hub)}
              disabled={isUpdating}
              onCheckedChange={handleToggleRequireAuthForPublicAIHub}
              ariaLabel={term(
                requireAuthForPublicAIHubProperty?.description ??
                  t("adminPanel.uiSettings.settings.requireAuthForPublicAIHub.label"),
              )}
              label={term(t("adminPanel.uiSettings.settings.requireAuthForPublicAIHub.label"))}
              description={
                requireAuthForPublicAIHubProperty?.description && term(requireAuthForPublicAIHubProperty.description)
              }
            />
            <SettingRow
              checked={Boolean(values.forward_client_headers_to_llm_api)}
              disabled={isUpdating}
              onCheckedChange={handleToggleForwardClientHeaders}
              ariaLabel={
                forwardClientHeadersProperty?.description ??
                t("adminPanel.uiSettings.settings.forwardClientHeaders.label")
              }
              label={t("adminPanel.uiSettings.settings.forwardClientHeaders.label")}
              description={
                forwardClientHeadersProperty?.description ??
                t("adminPanel.uiSettings.settings.forwardClientHeaders.description")
              }
            />
            <SettingRow
              checked={Boolean(values.forward_llm_provider_auth_headers)}
              disabled={isUpdating}
              onCheckedChange={handleToggleForwardLLMProviderAuthHeaders}
              ariaLabel={
                forwardLLMProviderAuthHeadersProperty?.description ??
                t("adminPanel.uiSettings.settings.forwardLLMProviderAuthHeaders.label")
              }
              label={t("adminPanel.uiSettings.settings.forwardLLMProviderAuthHeaders.label")}
              description={
                forwardLLMProviderAuthHeadersProperty?.description ??
                t("adminPanel.uiSettings.settings.forwardLLMProviderAuthHeaders.description")
              }
            />
            {enableProjectsUIProperty && (
              <SettingRow
                checked={Boolean(values.enable_projects_ui)}
                disabled={isUpdating}
                onCheckedChange={handleToggleEnableProjectsUI}
                ariaLabel={
                  enableProjectsUIProperty.description ?? t("adminPanel.uiSettings.settings.enableProjects.ariaLabel")
                }
                label={t("adminPanel.uiSettings.settings.enableProjects.label")}
                description={
                  enableProjectsUIProperty.description ?? t("adminPanel.uiSettings.settings.enableProjects.description")
                }
              />
            )}
            <SettingRow
              checked={Boolean(values.enable_chat_ui)}
              disabled={isUpdating}
              onCheckedChange={handleToggleEnableChatUI}
              ariaLabel={enableChatUIProperty?.description ?? t("adminPanel.uiSettings.settings.enableChat.ariaLabel")}
              label={t("adminPanel.uiSettings.settings.enableChat.label")}
              description={
                enableChatUIProperty?.description ?? t("adminPanel.uiSettings.settings.enableChat.description")
              }
            />

            <BackgroundChatSettings />

            <Separator />
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-1">
                <p className="text-sm font-medium text-foreground">Moyai</p>
                <p className="text-sm text-muted-foreground">
                  {moyaiUrl
                    ? t("adminPanel.uiSettings.moyai.connectedTo", { url: moyaiUrl })
                    : t("adminPanel.uiSettings.moyai.notConnected")}
                </p>
              </div>
              {moyaiUrl ? (
                <Button variant="outline" size="sm" disabled={isUpdating} onClick={handleDisconnectMoyai}>
                  {t("adminPanel.uiSettings.moyai.disconnect")}
                </Button>
              ) : (
                <a href={uiHref("moyai")} className="text-sm text-primary underline underline-offset-4">
                  {t("adminPanel.uiSettings.moyai.connectLink")}
                </a>
              )}
            </div>

            <Separator />
            <SettingRow
              checked={isAgentsDisabled}
              disabled={isUpdating}
              onCheckedChange={handleToggleDisableAgents}
              ariaLabel={disableAgentsProperty?.description ?? t("adminPanel.uiSettings.settings.disableAgents.label")}
              label={t("adminPanel.uiSettings.settings.disableAgents.label")}
              description={disableAgentsProperty?.description}
            />
            <SettingRow
              checked={Boolean(values.allow_agents_for_team_admins)}
              disabled={isUpdating || !isAgentsDisabled}
              onCheckedChange={handleToggleAllowAgentsTeamAdmins}
              ariaLabel={
                allowAgentsTeamAdminsProperty?.description ??
                t("adminPanel.uiSettings.settings.allowAgentsTeamAdmins.label")
              }
              label={t("adminPanel.uiSettings.settings.allowAgentsTeamAdmins.label")}
              description={allowAgentsTeamAdminsProperty?.description}
              indented
              muted={!isAgentsDisabled}
            />

            <Separator />
            <SettingRow
              checked={isVectorStoresDisabled}
              disabled={isUpdating}
              onCheckedChange={handleToggleDisableVectorStores}
              ariaLabel={
                disableVectorStoresProperty?.description ??
                t("adminPanel.uiSettings.settings.disableVectorStores.label")
              }
              label={t("adminPanel.uiSettings.settings.disableVectorStores.label")}
              description={disableVectorStoresProperty?.description}
            />
            <SettingRow
              checked={Boolean(values.allow_vector_stores_for_team_admins)}
              disabled={isUpdating || !isVectorStoresDisabled}
              onCheckedChange={handleToggleAllowVectorStoresTeamAdmins}
              ariaLabel={
                allowVectorStoresTeamAdminsProperty?.description ??
                t("adminPanel.uiSettings.settings.allowVectorStoresTeamAdmins.label")
              }
              label={t("adminPanel.uiSettings.settings.allowVectorStoresTeamAdmins.label")}
              description={allowVectorStoresTeamAdminsProperty?.description}
              indented
              muted={!isVectorStoresDisabled}
            />

            <Separator />
            <SettingRow
              checked={Boolean(values.scope_user_search_to_org)}
              disabled={isUpdating}
              onCheckedChange={handleToggleScopeUserSearch}
              ariaLabel={
                scopeUserSearchProperty?.description ?? t("adminPanel.uiSettings.settings.scopeUserSearch.label")
              }
              label={t("adminPanel.uiSettings.settings.scopeUserSearch.label")}
              description={
                scopeUserSearchProperty?.description ?? t("adminPanel.uiSettings.settings.scopeUserSearch.description")
              }
            />

            <Separator />
            <SettingRow
              checked={Boolean(values.disable_custom_api_keys)}
              disabled={isUpdating}
              onCheckedChange={handleToggleDisableCustomApiKeys}
              ariaLabel={
                disableCustomApiKeysProperty?.description ??
                t("adminPanel.uiSettings.settings.disableCustomApiKeys.label")
              }
              label={t("adminPanel.uiSettings.settings.disableCustomApiKeys.label")}
              description={
                disableCustomApiKeysProperty?.description ??
                t("adminPanel.uiSettings.settings.disableCustomApiKeys.description")
              }
            />

            <Separator />
            <PageVisibilitySettings
              enabledPagesInternalUsers={values.enabled_ui_pages_internal_users}
              enabledPagesPropertyDescription={enabledPagesProperty?.description}
              isUpdating={isUpdating}
              onUpdate={handleUpdatePageVisibility}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
