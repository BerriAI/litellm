import React, { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { ChevronRight, CircleMinus, Info, Plus, TriangleAlert, X } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Switch } from "@/components/ui/switch";
import { useFieldArray, useFormContext, useWatch } from "react-hook-form";
import { MCPServer, AUTH_TYPE } from "@/components/mcp_tools/types";
import {
  MountedFormField,
  useMountedName,
  type MountedFieldControlProps,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import { requiredRule } from "@/components/common_components/formRules";
import { Field, FieldLabel } from "@/components/ui/field";
import { invertedSwitchControl, switchControl, tagsControl, textControl } from "./mcpFieldRules";
import { listControl } from "./mcpFormStore";

interface MCPPermissionManagementProps {
  availableAccessGroups: string[];
  mcpServer: MCPServer | null;
  /**
   * The auth type as seen through the gate that mounts the auth_type field.
   * Callers pass undefined whenever that field is unmounted, because both
   * toggles below are mounted from this value and the payload only carries
   * what is mounted.
   */
  mountedAuthType: string | null | undefined;
}

const ClearableInput: React.FC<{
  control: MountedFieldControlProps;
  placeholder: string;
  clearLabel: string;
}> = ({ control, placeholder, clearLabel }) => {
  const text = textControl(control);
  return (
    <InputGroup className="rounded-lg">
      <InputGroupInput {...text} placeholder={placeholder} />
      {text.value !== "" && (
        <InputGroupAddon align="inline-end">
          <InputGroupButton size="icon-xs" aria-label={clearLabel} onClick={() => control.onChange("")}>
            <X />
          </InputGroupButton>
        </InputGroupAddon>
      )}
    </InputGroup>
  );
};

const StaticHeadersFieldArray: React.FC = () => {
  const { t } = useTranslation();
  const { control } = useFormContext<MountedFormValues>();
  const { fields, append, remove } = useFieldArray({ control: listControl(control), name: "static_headers" });
  useMountedName("static_headers");

  return (
    <div className="space-y-3">
      {fields.map((item, index) => (
        <div key={item.id} className="flex w-full items-baseline gap-4">
          <MountedFormField
            name={["static_headers", String(index), "header"]}
            className="flex-1"
            rules={{
              validate: {
                required: requiredRule(t("mcpServers.permissionManagement.staticHeaders.headerNameRequired")),
              },
            }}
          >
            {(headerControl) => (
              <ClearableInput
                control={headerControl}
                placeholder={t("mcpServers.permissionManagement.staticHeaders.headerNamePlaceholder")}
                clearLabel={t("mcpServers.permissionManagement.staticHeaders.clearHeaderName")}
              />
            )}
          </MountedFormField>
          <MountedFormField
            name={["static_headers", String(index), "value"]}
            className="flex-1"
            rules={{
              validate: {
                required: requiredRule(t("mcpServers.permissionManagement.staticHeaders.headerValueRequired")),
              },
            }}
          >
            {(valueControl) => (
              <ClearableInput
                control={valueControl}
                placeholder={t("mcpServers.permissionManagement.staticHeaders.headerValuePlaceholder")}
                clearLabel={t("mcpServers.permissionManagement.staticHeaders.clearHeaderValue")}
              />
            )}
          </MountedFormField>
          <CircleMinus
            onClick={() => remove(index)}
            className="size-4 text-muted-foreground hover:text-destructive cursor-pointer"
          />
        </div>
      ))}
      <Button variant="outline" className="w-full border-dashed" onClick={() => append({})}>
        <Plus />
        {t("mcpServers.permissionManagement.staticHeaders.add")}
      </Button>
    </div>
  );
};

const MCPPermissionManagement: React.FC<MCPPermissionManagementProps> = ({
  availableAccessGroups,
  mcpServer,
  mountedAuthType,
}) => {
  const { t } = useTranslation();
  const { setValue } = useFormContext<MountedFormValues>();
  const isOAuth2 = mountedAuthType === AUTH_TYPE.OAUTH2;
  const isNoneAuth = mountedAuthType === AUTH_TYPE.NONE || mountedAuthType == null;
  const watchedExtraHeaders = useWatch({ name: "extra_headers" });
  const hasAuthorizationHeader =
    Array.isArray(watchedExtraHeaders) &&
    watchedExtraHeaders.some((h) => typeof h === "string" && h.toLowerCase() === "authorization");
  // Two distinct, independent opt-ins:
  //   - delegate_auth_to_upstream: oauth2 servers only (PKCE passthrough —
  //     bypass LiteLLM admission).
  //   - oauth_passthrough: auth_type=none + Authorization in extra_headers
  //     (OAuth pass-through: proxy upstream oauth-protected-resource, emit 401
  //     challenges, propagate upstream 401/403).
  // Kept as separate flags so neither silently implies the other and existing
  // oauth2 servers can't regress into pass-through behavior.
  const canEnableOAuthPassthrough = isNoneAuth && hasAuthorizationHeader;
  const watchedDelegateAuth = useWatch({ name: "delegate_auth_to_upstream" });
  const watchedPublicInternet = useWatch({ name: "available_on_public_internet" });
  const showInternalDelegatePkceWarning = isOAuth2 && watchedDelegateAuth === true && watchedPublicInternet === false;

  // Set initial values when mcpServer changes
  useEffect(() => {
    if (mcpServer) {
      if (mcpServer.static_headers) {
        const staticHeaders = Object.entries(mcpServer.static_headers).map(([header, value]) => ({
          header,
          value: value != null ? String(value) : "",
        }));
        setValue("static_headers", staticHeaders);
      }
      if (Array.isArray(mcpServer.env_vars) && mcpServer.env_vars.length > 0) {
        setValue(
          "env_vars",
          mcpServer.env_vars.map((entry) => ({
            name: entry.name,
            value: entry.value ?? "",
            scope: entry.scope ?? "global",
            description: entry.description ?? "",
          })),
        );
      }
      if (typeof mcpServer.allow_all_keys === "boolean") {
        setValue("allow_all_keys", mcpServer.allow_all_keys);
      }
      if (typeof mcpServer.available_on_public_internet === "boolean") {
        setValue("available_on_public_internet", mcpServer.available_on_public_internet);
      }
      if (typeof mcpServer.delegate_auth_to_upstream === "boolean") {
        setValue("delegate_auth_to_upstream", mcpServer.delegate_auth_to_upstream);
      }
      if (typeof mcpServer.oauth_passthrough === "boolean") {
        setValue("oauth_passthrough", mcpServer.oauth_passthrough);
      }
    } else {
      setValue("allow_all_keys", false);
      setValue("available_on_public_internet", true);
      setValue("delegate_auth_to_upstream", false);
      setValue("oauth_passthrough", false);
    }
  }, [mcpServer, setValue]);

  // delegate_auth_to_upstream is only honored server-side for oauth2 servers.
  // Force it back to false whenever the user switches away from oauth2 so a
  // stale toggle value doesn't get persisted unexpectedly.
  useEffect(() => {
    if (!isOAuth2) {
      setValue("delegate_auth_to_upstream", false);
    }
  }, [isOAuth2, setValue]);

  // oauth_passthrough is only honored for auth_type=none servers that forward
  // Authorization upstream. Force it back to false otherwise.
  useEffect(() => {
    if (!canEnableOAuthPassthrough) {
      setValue("oauth_passthrough", false);
    }
  }, [canEnableOAuthPassthrough, setValue]);

  return (
    <Collapsible className="bg-muted border border-border rounded-lg">
      <CollapsibleTrigger className="group flex w-full items-center justify-between gap-4 p-4 text-left">
        <span className="flex items-center">
          <span className="flex items-center space-x-2">
            <span className="w-2 h-2 bg-info rounded-full"></span>
            <span className="text-lg font-semibold text-foreground">{t("mcpServers.permissionManagement.title")}</span>
          </span>
          <span className="text-sm text-muted-foreground ml-4">{t("mcpServers.permissionManagement.subtitle")}</span>
        </span>
        <ChevronRight className="size-4 shrink-0 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
      </CollapsibleTrigger>
      <CollapsibleContent keepMounted className="px-4 pb-4">
        <div className="space-y-6 pt-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="text-sm font-medium text-foreground flex items-center">
                {t("mcpServers.permissionManagement.allowAllKeys.label")}
                <SimpleTooltip content={t("mcpServers.permissionManagement.allowAllKeys.tooltip")}>
                  <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                </SimpleTooltip>
              </span>
              <p className="text-sm text-muted-foreground mt-1">
                {t("mcpServers.permissionManagement.allowAllKeys.description")}
              </p>
            </div>
            <MountedFormField name="allow_all_keys" defaultValue={mcpServer?.allow_all_keys ?? false} className="mb-0">
              {(control) => (
                <Switch
                  aria-label={t("mcpServers.permissionManagement.allowAllKeys.label")}
                  {...switchControl(control)}
                />
              )}
            </MountedFormField>
          </div>

          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="text-sm font-medium text-foreground flex items-center">
                {t("mcpServers.permissionManagement.internalNetworkOnly.label")}
                <SimpleTooltip content={t("mcpServers.permissionManagement.internalNetworkOnly.tooltip")}>
                  <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                </SimpleTooltip>
              </span>
              <p className="text-sm text-muted-foreground mt-1">
                {t("mcpServers.permissionManagement.internalNetworkOnly.description")}
              </p>
            </div>
            <MountedFormField name="available_on_public_internet" defaultValue={true} className="mb-0">
              {(control) => (
                <Switch
                  aria-label={t("mcpServers.permissionManagement.internalNetworkOnly.label")}
                  {...invertedSwitchControl(control)}
                />
              )}
            </MountedFormField>
          </div>

          {isOAuth2 && (
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-sm font-medium text-foreground flex items-center">
                  {t("mcpServers.permissionManagement.delegateAuth.label")}
                  <SimpleTooltip content={t("mcpServers.permissionManagement.delegateAuth.tooltip")}>
                    <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                  </SimpleTooltip>
                </span>
                <p className="text-sm text-muted-foreground mt-1">
                  {t("mcpServers.permissionManagement.delegateAuth.description")}
                </p>
              </div>
              <MountedFormField
                name="delegate_auth_to_upstream"
                defaultValue={mcpServer?.delegate_auth_to_upstream ?? false}
                className="mb-0"
              >
                {(control) => (
                  <Switch
                    aria-label={t("mcpServers.permissionManagement.delegateAuth.label")}
                    {...switchControl(control)}
                  />
                )}
              </MountedFormField>
            </div>
          )}

          {canEnableOAuthPassthrough && (
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-sm font-medium text-foreground flex items-center">
                  {t("mcpServers.permissionManagement.oauthPassthrough.label")}
                  <SimpleTooltip content={t("mcpServers.permissionManagement.oauthPassthrough.tooltip")}>
                    <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                  </SimpleTooltip>
                </span>
                <p className="text-sm text-muted-foreground mt-1">
                  {t("mcpServers.permissionManagement.oauthPassthrough.description")}
                </p>
              </div>
              <MountedFormField
                name="oauth_passthrough"
                defaultValue={mcpServer?.oauth_passthrough ?? false}
                className="mb-0"
              >
                {(control) => (
                  <Switch
                    aria-label={t("mcpServers.permissionManagement.oauthPassthrough.label")}
                    {...switchControl(control)}
                  />
                )}
              </MountedFormField>
            </div>
          )}

          {showInternalDelegatePkceWarning && (
            <Alert variant="warning" className="mb-2">
              <TriangleAlert />
              <AlertTitle>{t("mcpServers.permissionManagement.internalDelegateWarning.title")}</AlertTitle>
              <AlertDescription>
                {t("mcpServers.permissionManagement.internalDelegateWarning.description")}
              </AlertDescription>
            </Alert>
          )}

          <MountedFormField
            label={
              <span className="text-sm font-medium text-foreground flex items-center">
                {t("mcpServers.permissionManagement.accessGroups.label")}
                <SimpleTooltip content={t("mcpServers.permissionManagement.accessGroups.tooltip")}>
                  <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                </SimpleTooltip>
              </span>
            }
            name="mcp_access_groups"
            className="mb-4"
          >
            {(control) => (
              <MultiSelect
                {...tagsControl(control, t)}
                options={availableAccessGroups.map((group) => ({ label: group, value: group }))}
                placeholder={t("mcpServers.permissionManagement.accessGroups.placeholder")}
                className="rounded-lg"
              />
            )}
          </MountedFormField>

          <MountedFormField
            label={
              <span className="text-sm font-medium text-foreground flex items-center">
                {t("mcpServers.permissionManagement.extraHeaders.label")}
                <SimpleTooltip content={t("mcpServers.permissionManagement.extraHeaders.tooltip")}>
                  <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                </SimpleTooltip>
                {mcpServer?.extra_headers && mcpServer.extra_headers.length > 0 && (
                  <span className="ml-2 text-xs bg-info/15 text-info px-2 py-1 rounded-full">
                    {t("mcpServers.permissionManagement.extraHeaders.configuredCount", {
                      count: mcpServer.extra_headers.length,
                    })}
                  </span>
                )}
              </span>
            }
            name="extra_headers"
          >
            {(control) => (
              <MultiSelect
                {...tagsControl(control, t)}
                placeholder={
                  mcpServer?.extra_headers && mcpServer.extra_headers.length > 0
                    ? t("mcpServers.permissionManagement.extraHeaders.currentlyPlaceholder", {
                        headers: mcpServer.extra_headers.join(", "),
                      })
                    : t("mcpServers.permissionManagement.extraHeaders.placeholder")
                }
                className="rounded-lg"
              />
            )}
          </MountedFormField>

          <Field>
            <FieldLabel>
              <span className="text-sm font-medium text-foreground flex items-center">
                {t("mcpServers.permissionManagement.staticHeaders.label")}
                <SimpleTooltip content={t("mcpServers.permissionManagement.staticHeaders.tooltip")}>
                  <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                </SimpleTooltip>
              </span>
            </FieldLabel>
            <StaticHeadersFieldArray />
          </Field>
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

export default MCPPermissionManagement;
