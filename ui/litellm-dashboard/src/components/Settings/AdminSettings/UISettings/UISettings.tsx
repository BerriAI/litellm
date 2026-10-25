"use client";

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
import { useLanguage } from "@/contexts/LanguageContext";
import { isLanguage, resolveLanguage } from "@/i18n/language";
import { useId } from "react";
import { isProxyAdminRole } from "@/utils/roles";
import TranslatedMessage from "@/components/shared/TranslatedMessage";

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
  const { t } = useLanguage();
  const defaultLanguageId = useId();
  const { accessToken, userRole, isViewOnly } = useAuthorized();
  const { data, isLoading, isError, refetch } = useUISettings();
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
  const languageConfigOwned = data?.source?.default_ui_language === "config";
  // 会话角色可能将只读管理员映射为 Admin，必须同时检查独立的只读标记
  const canEditDefaultLanguage = isProxyAdminRole(userRole ?? "") && !isViewOnly;
  const languageEditable =
    canEditDefaultLanguage && !languageConfigOwned && Boolean(schema?.properties?.default_ui_language);
  const usageRange =
    relativeTimeOptions.find((option) => option.settingValue === values.usage_default_time_range)?.settingValue ??
    "last_7_days";
  const usageRangeConfigOwned = data?.source?.usage_default_time_range === "config";
  const moyaiUrl = (values.moyai_url as string | undefined) ?? null;
  const isDisabledForInternalUsers = Boolean(values.disable_model_add_for_internal_users);
  const isDisabledTeamAdminDeleteTeamUser = Boolean(values.disable_team_admin_delete_team_user);
  const isAgentsDisabled = Boolean(values.disable_agents_for_internal_users);
  const isVectorStoresDisabled = Boolean(values.disable_vector_stores_for_internal_users);

  const handleDefaultLanguageChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const value = event.target.value;
    if (!isLanguage(value)) return;
    updateSettings(
      { default_ui_language: value },
      {
        // 固定通知在 Toaster 内读取当前语言，不捕获请求发起时的语言。
        onSuccess: () => toast.success(<TranslatedMessage message="UI settings updated successfully" />),
        onError: () => toast.error(<TranslatedMessage message="Could not update UI settings" />),
      },
    );
  };

  const handleToggle = (checked: boolean) => {
    updateSettings(
      { disable_model_add_for_internal_users: checked },
      {
        onSuccess: () => {
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
        toast.success(<TranslatedMessage message="Page visibility settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully. Refreshing page..." />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully. Refreshing page..." />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          toast.success(<TranslatedMessage message="UI settings updated successfully" />);
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
          <h3>{t("UI Settings")}</h3>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div role="status" aria-label={t("Loading UI settings")} className="space-y-3">
            <Skeleton className="h-5 w-72" />
            <Skeleton className="h-16 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        ) : isError ? (
          <Alert variant="error">
            <AlertTitle>{t("Could not load UI settings")}</AlertTitle>
            <AlertDescription>{t("Check your connection and retry. (UI_SETTINGS_READ_FAILED)")}</AlertDescription>
            <Button variant="outline" onClick={() => void refetch()}>
              {t("Retry")}
            </Button>
          </Alert>
        ) : (
          <div className="space-y-6">
            {/* 全站默认设置沿用后端管理员校验，配置文件管理的值只读 */}
            <div className="space-y-2">
              <label htmlFor={defaultLanguageId} className="block text-sm font-medium">
                {t("Default UI language")}
              </label>
              <p id={`${defaultLanguageId}-help`} className="text-sm text-muted-foreground">
                {t("Used when a user has not chosen a language. Personal choices take priority.")}
              </p>
              <select
                id={defaultLanguageId}
                aria-describedby={`${defaultLanguageId}-help`}
                value={resolveLanguage(null, values.default_ui_language)}
                disabled={isUpdating || !languageEditable}
                onChange={handleDefaultLanguageChange}
                className="h-11 rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-2 focus-visible:outline-ring md:h-9"
              >
                <option value="en" lang="en">
                  English
                </option>
                <option value="zh-CN" lang="zh-CN">
                  简体中文
                </option>
              </select>
              {languageConfigOwned && (
                <p className="text-xs text-muted-foreground">{t("Managed by configuration file")}</p>
              )}
              {!schema?.properties?.default_ui_language && (
                <p className="text-xs text-muted-foreground">
                  {t("Upgrade the proxy to configure the default language.")}
                </p>
              )}
              {isUpdating && (
                <p role="status" className="text-sm text-muted-foreground">
                  {t("Saving...")}
                </p>
              )}
            </div>
            {/* 后端 schema 描述可能由部署方自定义，按原文展示。 */}
            {schema?.description && <p className="text-sm text-foreground">{schema.description}</p>}
            {updateError && (
              <Alert variant="error">
                <AlertTitle>{t("Could not update UI settings")}</AlertTitle>
                <AlertDescription>
                  {t("Your change was not saved. Check your connection and try again. (UI_SETTINGS_WRITE_FAILED)")}
                </AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="usage-default-time-range">{t("Default Usage time range")}</Label>
              <p id="usage-default-time-range-help" className="text-sm text-muted-foreground">
                {t(
                  "The starting range for everyone opening Usage. Shared links and manual date selections take precedence. Calendar ranges use your browser’s timezone.",
                )}
              </p>
              <Select
                value={usageRange}
                disabled={isUpdating || usageRangeConfigOwned}
                items={relativeTimeOptions.map((option) => ({ value: option.settingValue, label: t(option.label) }))}
                onValueChange={(value) => {
                  if (value === null) return;
                  updateSettings(
                    { usage_default_time_range: value },
                    {
                      onSuccess: () => toast.success(<TranslatedMessage message="UI settings updated successfully" />),
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
                      {t(option.label)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {usageRangeConfigOwned && (
                <p className="text-sm text-muted-foreground">{t("Managed by the deployment configuration.")}</p>
              )}
            </div>
            <SettingRow
              checked={isDisabledForInternalUsers}
              disabled={isUpdating}
              onCheckedChange={handleToggle}
              ariaLabel={property?.description ?? t("Disable model add for internal users")}
              label={t("Disable model add for internal users")}
              description={property?.description}
            />
            <SettingRow
              checked={isDisabledTeamAdminDeleteTeamUser}
              disabled={isUpdating}
              onCheckedChange={handleToggleTeamAdminDelete}
              ariaLabel={disableTeamAdminDeleteProperty?.description ?? t("Disable team admin delete team user")}
              label={t("Disable team admin delete team user")}
              description={disableTeamAdminDeleteProperty?.description}
            />
            <SettingRow
              checked={Boolean(values.require_auth_for_public_ai_hub)}
              disabled={isUpdating}
              onCheckedChange={handleToggleRequireAuthForPublicAIHub}
              ariaLabel={
                requireAuthForPublicAIHubProperty?.description ?? t("Require authentication for public AI Hub")
              }
              label={t("Require authentication for public AI Hub")}
              description={requireAuthForPublicAIHubProperty?.description}
            />
            <SettingRow
              checked={Boolean(values.forward_client_headers_to_llm_api)}
              disabled={isUpdating}
              onCheckedChange={handleToggleForwardClientHeaders}
              ariaLabel={forwardClientHeadersProperty?.description ?? t("Forward client headers to LLM API")}
              label={t("Forward client headers to LLM API")}
              description={
                forwardClientHeadersProperty?.description ??
                t(
                  "Forwards client headers (Authorization, anthropic-beta, and x-* custom headers) to the upstream LLM. Enable for Claude Code with a Max subscription (forwards the OAuth token) or to pass custom/tracing headers through to the provider. Independent of the BYOK toggle — enable only the one(s) you need.",
                )
              }
            />
            <SettingRow
              checked={Boolean(values.forward_llm_provider_auth_headers)}
              disabled={isUpdating}
              onCheckedChange={handleToggleForwardLLMProviderAuthHeaders}
              ariaLabel={forwardLLMProviderAuthHeadersProperty?.description ?? t("Forward LLM provider auth headers")}
              label={t("Forward LLM provider auth headers")}
              description={
                forwardLLMProviderAuthHeadersProperty?.description ??
                t(
                  "Forwards provider auth headers (x-api-key, x-goog-api-key, api-key, ocp-apim-subscription-key) to the upstream LLM, overriding any deployment-configured key for that request. Enable for Claude Code BYOK (clients bring their own API key). Independent of the client-headers toggle — enable only the one(s) you need.",
                )
              }
            />
            {enableProjectsUIProperty && (
              <SettingRow
                checked={Boolean(values.enable_projects_ui)}
                disabled={isUpdating}
                onCheckedChange={handleToggleEnableProjectsUI}
                ariaLabel={enableProjectsUIProperty.description ?? t("Enable Projects UI")}
                label={t("[BETA] Enable Projects (page will refresh)")}
                description={
                  enableProjectsUIProperty.description ??
                  t("If enabled, shows the Projects feature in the UI sidebar and the project field in key management.")
                }
              />
            )}
            <SettingRow
              checked={Boolean(values.enable_chat_ui)}
              disabled={isUpdating}
              onCheckedChange={handleToggleEnableChatUI}
              ariaLabel={enableChatUIProperty?.description ?? t("Enable Chat page")}
              label={t("[BETA] Enable Chat page (page will refresh)")}
              description={
                enableChatUIProperty?.description ??
                t(
                  "If enabled, shows the Chat page in the UI sidebar, letting users chat with an LLM and connect their own MCP server credentials via OAuth.",
                )
              }
            />

            <Separator />
            <div className="flex items-start justify-between gap-3">
              <div className="space-y-1">
                <p className="text-sm font-medium text-foreground">Moyai</p>
                <p className="text-sm text-muted-foreground">
                  {moyaiUrl ? t("Connected to {url}", { url: moyaiUrl }) : t("Not connected")}
                </p>
              </div>
              {moyaiUrl ? (
                <Button variant="outline" size="sm" disabled={isUpdating} onClick={handleDisconnectMoyai}>
                  {t("Disconnect")}
                </Button>
              ) : (
                <a href={uiHref("moyai")} className="text-sm text-primary underline underline-offset-4">
                  {t("Connect from the Moyai page")}
                </a>
              )}
            </div>

            <Separator />
            <SettingRow
              checked={isAgentsDisabled}
              disabled={isUpdating}
              onCheckedChange={handleToggleDisableAgents}
              ariaLabel={disableAgentsProperty?.description ?? t("Disable agents for internal users")}
              label={t("Disable agents for internal users")}
              description={disableAgentsProperty?.description}
            />
            <SettingRow
              checked={Boolean(values.allow_agents_for_team_admins)}
              disabled={isUpdating || !isAgentsDisabled}
              onCheckedChange={handleToggleAllowAgentsTeamAdmins}
              ariaLabel={allowAgentsTeamAdminsProperty?.description ?? t("Allow agents for team admins")}
              label={t("Allow agents for team admins")}
              description={allowAgentsTeamAdminsProperty?.description}
              indented
              muted={!isAgentsDisabled}
            />

            <Separator />
            <SettingRow
              checked={isVectorStoresDisabled}
              disabled={isUpdating}
              onCheckedChange={handleToggleDisableVectorStores}
              ariaLabel={disableVectorStoresProperty?.description ?? t("Disable vector stores for internal users")}
              label={t("Disable vector stores for internal users")}
              description={disableVectorStoresProperty?.description}
            />
            <SettingRow
              checked={Boolean(values.allow_vector_stores_for_team_admins)}
              disabled={isUpdating || !isVectorStoresDisabled}
              onCheckedChange={handleToggleAllowVectorStoresTeamAdmins}
              ariaLabel={allowVectorStoresTeamAdminsProperty?.description ?? t("Allow vector stores for team admins")}
              label={t("Allow vector stores for team admins")}
              description={allowVectorStoresTeamAdminsProperty?.description}
              indented
              muted={!isVectorStoresDisabled}
            />

            <Separator />
            <SettingRow
              checked={Boolean(values.scope_user_search_to_org)}
              disabled={isUpdating}
              onCheckedChange={handleToggleScopeUserSearch}
              ariaLabel={scopeUserSearchProperty?.description ?? t("Scope user search to organization")}
              label={t("Scope user search to organization")}
              description={
                scopeUserSearchProperty?.description ??
                t(
                  "If enabled, the user search endpoint restricts results by organization. When off, any authenticated user can search all users.",
                )
              }
            />

            <Separator />
            <SettingRow
              checked={Boolean(values.disable_custom_api_keys)}
              disabled={isUpdating}
              onCheckedChange={handleToggleDisableCustomApiKeys}
              ariaLabel={disableCustomApiKeysProperty?.description ?? t("Disable custom Virtual key values")}
              label={t("Disable custom Virtual key values")}
              description={
                disableCustomApiKeysProperty?.description ??
                t("If true, users cannot specify custom key values. All keys must be auto-generated.")
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
