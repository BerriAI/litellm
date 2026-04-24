"use client";

import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useUpdateUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUpdateUISettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import NotificationManager from "@/components/molecules/notifications_manager";
import PageVisibilitySettings from "./PageVisibilitySettings";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";

function SettingRow({
  checked,
  disabled,
  onCheckedChange,
  ariaLabel,
  title,
  description,
  titleMuted,
  indent,
}: {
  checked: boolean;
  disabled: boolean;
  onCheckedChange: (checked: boolean) => void;
  ariaLabel?: string;
  title: string;
  description?: string;
  titleMuted?: boolean;
  indent?: boolean;
}) {
  return (
    <div className={`flex items-start gap-4 ${indent ? "ml-8" : ""}`}>
      <Switch
        checked={checked}
        disabled={disabled}
        onCheckedChange={onCheckedChange}
        aria-label={ariaLabel}
        className="mt-1"
      />
      <div className="flex flex-col gap-1">
        <span className={`font-semibold ${titleMuted ? "text-muted-foreground" : ""}`}>{title}</span>
        {description && <span className="text-sm text-muted-foreground">{description}</span>}
      </div>
    </div>
  );
}

export default function UISettings() {
  const { accessToken } = useAuthorized();
  const { data, isLoading, isError, error } = useUISettings();
  const { mutate: updateSettings, isPending: isUpdating, error: updateError } = useUpdateUISettings(accessToken);

  const schema = data?.field_schema;
  const property = schema?.properties?.disable_model_add_for_internal_users;
  const disableTeamAdminDeleteProperty = schema?.properties?.disable_team_admin_delete_team_user;
  const requireAuthForPublicAIHubProperty = schema?.properties?.require_auth_for_public_ai_hub;
  const forwardClientHeadersProperty = schema?.properties?.forward_client_headers_to_llm_api;
  const forwardLLMProviderAuthHeadersProperty =
    schema?.properties?.forward_llm_provider_auth_headers;
  const enableProjectsUIProperty = schema?.properties?.enable_projects_ui;
  const enabledPagesProperty = schema?.properties?.enabled_ui_pages_internal_users;
  const disableAgentsProperty = schema?.properties?.disable_agents_for_internal_users;
  const allowAgentsTeamAdminsProperty = schema?.properties?.allow_agents_for_team_admins;
  const disableVectorStoresProperty = schema?.properties?.disable_vector_stores_for_internal_users;
  const allowVectorStoresTeamAdminsProperty = schema?.properties?.allow_vector_stores_for_team_admins;
  const scopeUserSearchProperty = schema?.properties?.scope_user_search_to_org;
  const disableCustomApiKeysProperty = schema?.properties?.disable_custom_api_keys;
  const values = data?.values ?? {};
  const isDisabledForInternalUsers = Boolean(values.disable_model_add_for_internal_users);
  const isDisabledTeamAdminDeleteTeamUser = Boolean(values.disable_team_admin_delete_team_user);
  const isAgentsDisabled = Boolean(values.disable_agents_for_internal_users);
  const isVectorStoresDisabled = Boolean(values.disable_vector_stores_for_internal_users);

  const toggleField = (
    key: string,
    successMsg = "UI settings updated successfully",
    reload = false,
  ) =>
    (checked: boolean) => {
      updateSettings(
        { [key]: checked },
        {
          onSuccess: () => {
            NotificationManager.success(
              reload ? `${successMsg}. Refreshing page...` : successMsg,
            );
            if (reload) setTimeout(() => window.location.reload(), 1000);
          },
          onError: (err) => {
            NotificationManager.fromBackend(err);
          },
        },
      );
    };

  const handleUpdatePageVisibility = (settings: { enabled_ui_pages_internal_users: string[] | null }) => {
    updateSettings(settings, {
      onSuccess: () => {
        NotificationManager.success("Page visibility settings updated successfully");
      },
      onError: (err) => {
        NotificationManager.fromBackend(err);
      },
    });
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">UI Settings</h3>
      {isLoading ? (
        <div className="space-y-2">
          <Skeleton className="h-6 w-full" />
          <Skeleton className="h-6 w-3/4" />
          <Skeleton className="h-6 w-1/2" />
        </div>
      ) : isError ? (
        <Alert variant="destructive">
          <AlertTitle>Could not load UI settings</AlertTitle>
          {error instanceof Error ? (
            <AlertDescription>{error.message}</AlertDescription>
          ) : null}
        </Alert>
      ) : (
        <div className="flex flex-col gap-6 w-full">
          {schema?.description && (
            <p className="mb-0">{schema.description}</p>
          )}

          {updateError && (
            <Alert variant="destructive">
              <AlertTitle>Could not update UI settings</AlertTitle>
              {updateError instanceof Error ? (
                <AlertDescription>{updateError.message}</AlertDescription>
              ) : null}
            </Alert>
          )}

          <SettingRow
            checked={isDisabledForInternalUsers}
            disabled={isUpdating}
            onCheckedChange={toggleField("disable_model_add_for_internal_users")}
            ariaLabel={property?.description ?? "Disable model add for internal users"}
            title="Disable model add for internal users"
            description={property?.description}
          />

          <SettingRow
            checked={isDisabledTeamAdminDeleteTeamUser}
            disabled={isUpdating}
            onCheckedChange={toggleField("disable_team_admin_delete_team_user")}
            ariaLabel={disableTeamAdminDeleteProperty?.description ?? "Disable team admin delete team user"}
            title="Disable team admin delete team user"
            description={disableTeamAdminDeleteProperty?.description}
          />

          <SettingRow
            checked={Boolean(values.require_auth_for_public_ai_hub)}
            disabled={isUpdating}
            onCheckedChange={toggleField("require_auth_for_public_ai_hub")}
            ariaLabel={
              requireAuthForPublicAIHubProperty?.description ??
              "Require authentication for public AI Hub"
            }
            title="Require authentication for public AI Hub"
            description={requireAuthForPublicAIHubProperty?.description}
          />

          <SettingRow
            checked={Boolean(values.forward_client_headers_to_llm_api)}
            disabled={isUpdating}
            onCheckedChange={toggleField("forward_client_headers_to_llm_api")}
            ariaLabel={forwardClientHeadersProperty?.description ?? "Forward client headers to LLM API"}
            title="Forward client headers to LLM API"
            description={
              forwardClientHeadersProperty?.description ??
              "Forwards client headers (Authorization, anthropic-beta, and x-* custom headers) to the upstream LLM. Enable for Claude Code with a Max subscription (forwards the OAuth token) or to pass custom/tracing headers through to the provider. Independent of the BYOK toggle — enable only the one(s) you need."
            }
          />

          <SettingRow
            checked={Boolean(values.forward_llm_provider_auth_headers)}
            disabled={isUpdating}
            onCheckedChange={toggleField("forward_llm_provider_auth_headers")}
            ariaLabel={
              forwardLLMProviderAuthHeadersProperty?.description ?? "Forward LLM provider auth headers"
            }
            title="Forward LLM provider auth headers"
            description={
              forwardLLMProviderAuthHeadersProperty?.description ??
              "Forwards provider auth headers (x-api-key, x-goog-api-key, api-key, ocp-apim-subscription-key) to the upstream LLM, overriding any deployment-configured key for that request. Enable for Claude Code BYOK (clients bring their own API key). Independent of the client-headers toggle — enable only the one(s) you need."
            }
          />

          <SettingRow
            checked={Boolean(values.enable_projects_ui)}
            disabled={isUpdating}
            onCheckedChange={toggleField("enable_projects_ui", "UI settings updated successfully", true)}
            ariaLabel={enableProjectsUIProperty?.description ?? "Enable Projects UI"}
            title="[BETA] Enable Projects (page will refresh)"
            description={
              enableProjectsUIProperty?.description ??
              "If enabled, shows the Projects feature in the UI sidebar and the project field in key management."
            }
          />

          <Separator />

          {/* Agents access control */}
          <SettingRow
            checked={isAgentsDisabled}
            disabled={isUpdating}
            onCheckedChange={toggleField("disable_agents_for_internal_users")}
            ariaLabel={disableAgentsProperty?.description ?? "Disable agents for internal users"}
            title="Disable agents for internal users"
            description={disableAgentsProperty?.description}
          />

          <SettingRow
            checked={Boolean(values.allow_agents_for_team_admins)}
            disabled={isUpdating || !isAgentsDisabled}
            onCheckedChange={toggleField("allow_agents_for_team_admins")}
            ariaLabel={allowAgentsTeamAdminsProperty?.description ?? "Allow agents for team admins"}
            title="Allow agents for team admins"
            description={allowAgentsTeamAdminsProperty?.description}
            titleMuted={!isAgentsDisabled}
            indent
          />

          <Separator />

          {/* Vector Stores access control */}
          <SettingRow
            checked={isVectorStoresDisabled}
            disabled={isUpdating}
            onCheckedChange={toggleField("disable_vector_stores_for_internal_users")}
            ariaLabel={disableVectorStoresProperty?.description ?? "Disable vector stores for internal users"}
            title="Disable vector stores for internal users"
            description={disableVectorStoresProperty?.description}
          />

          <SettingRow
            checked={Boolean(values.allow_vector_stores_for_team_admins)}
            disabled={isUpdating || !isVectorStoresDisabled}
            onCheckedChange={toggleField("allow_vector_stores_for_team_admins")}
            ariaLabel={allowVectorStoresTeamAdminsProperty?.description ?? "Allow vector stores for team admins"}
            title="Allow vector stores for team admins"
            description={allowVectorStoresTeamAdminsProperty?.description}
            titleMuted={!isVectorStoresDisabled}
            indent
          />

          <Separator />

          {/* Scope user search to organization */}
          <SettingRow
            checked={Boolean(values.scope_user_search_to_org)}
            disabled={isUpdating}
            onCheckedChange={toggleField("scope_user_search_to_org")}
            ariaLabel={scopeUserSearchProperty?.description ?? "Scope user search to organization"}
            title="Scope user search to organization"
            description={
              scopeUserSearchProperty?.description ??
              "If enabled, the user search endpoint restricts results by organization. When off, any authenticated user can search all users."
            }
          />

          <Separator />

          {/* Disable custom Virtual key values */}
          <SettingRow
            checked={Boolean(values.disable_custom_api_keys)}
            disabled={isUpdating}
            onCheckedChange={toggleField("disable_custom_api_keys")}
            ariaLabel={disableCustomApiKeysProperty?.description ?? "Disable custom Virtual key values"}
            title="Disable custom Virtual key values"
            description={
              disableCustomApiKeysProperty?.description ??
              "If true, users cannot specify custom key values. All keys must be auto-generated."
            }
          />

          <Separator />

          {/* Page Visibility for Internal Users */}
          <PageVisibilitySettings
            enabledPagesInternalUsers={values.enabled_ui_pages_internal_users}
            enabledPagesPropertyDescription={enabledPagesProperty?.description}
            isUpdating={isUpdating}
            onUpdate={handleUpdatePageVisibility}
          />
        </div>
      )}
    </Card>
  );
}
