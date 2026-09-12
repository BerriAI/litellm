import React, { useEffect } from "react";
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
            rules={{ validate: { required: requiredRule("Header name is required") } }}
          >
            {(headerControl) => (
              <ClearableInput
                control={headerControl}
                placeholder="Header name (e.g., X-API-Key)"
                clearLabel="Clear header name"
              />
            )}
          </MountedFormField>
          <MountedFormField
            name={["static_headers", String(index), "value"]}
            className="flex-1"
            rules={{ validate: { required: requiredRule("Header value is required") } }}
          >
            {(valueControl) => (
              <ClearableInput control={valueControl} placeholder="Header value" clearLabel="Clear header value" />
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
        Add Static Header
      </Button>
    </div>
  );
};

const MCPPermissionManagement: React.FC<MCPPermissionManagementProps> = ({
  availableAccessGroups,
  mcpServer,
  mountedAuthType,
}) => {
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
            <span className="text-lg font-semibold text-foreground">Permission Management / Access Control</span>
          </span>
          <span className="text-sm text-muted-foreground ml-4">
            Configure access permissions and security settings (Optional)
          </span>
        </span>
        <ChevronRight className="size-4 shrink-0 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
      </CollapsibleTrigger>
      <CollapsibleContent keepMounted className="px-4 pb-4">
        <div className="space-y-6 pt-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="text-sm font-medium text-foreground flex items-center">
                Allow All LiteLLM Keys
                <SimpleTooltip content="When enabled, every API key can access this MCP server.">
                  <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                </SimpleTooltip>
              </span>
              <p className="text-sm text-muted-foreground mt-1">
                Enable if this server should be &quot;public&quot; to all keys.
              </p>
            </div>
            <MountedFormField name="allow_all_keys" defaultValue={mcpServer?.allow_all_keys ?? false} className="mb-0">
              {(control) => <Switch aria-label="Allow All LiteLLM Keys" {...switchControl(control)} />}
            </MountedFormField>
          </div>

          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="text-sm font-medium text-foreground flex items-center">
                Internal network only
                <SimpleTooltip content="When on, only requests from within your internal network are accepted. Turn off to allow external clients (other clusters, ChatGPT, etc). API key authentication is always required regardless of this setting.">
                  <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                </SimpleTooltip>
              </span>
              <p className="text-sm text-muted-foreground mt-1">
                Turn on to restrict access to callers within your internal network only.
              </p>
            </div>
            <MountedFormField name="available_on_public_internet" defaultValue={true} className="mb-0">
              {(control) => <Switch aria-label="Internal network only" {...invertedSwitchControl(control)} />}
            </MountedFormField>
          </div>

          {isOAuth2 && (
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-sm font-medium text-foreground flex items-center">
                  Delegate auth to upstream (PKCE passthrough)
                  <SimpleTooltip content="When on, LiteLLM skips its own API key/SSO check for this server and lets the client complete PKCE directly with the upstream MCP server. Only honored when Auth Type is oauth2. No spend tracking or per-key rate limiting will run on this route.">
                    <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                  </SimpleTooltip>
                </span>
                <p className="text-sm text-muted-foreground mt-1">
                  Bypass LiteLLM auth so clients authenticate directly with the upstream OAuth MCP server.
                </p>
              </div>
              <MountedFormField
                name="delegate_auth_to_upstream"
                defaultValue={mcpServer?.delegate_auth_to_upstream ?? false}
                className="mb-0"
              >
                {(control) => (
                  <Switch aria-label="Delegate auth to upstream (PKCE passthrough)" {...switchControl(control)} />
                )}
              </MountedFormField>
            </div>
          )}

          {canEnableOAuthPassthrough && (
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-sm font-medium text-foreground flex items-center">
                  OAuth pass-through
                  <SimpleTooltip content="When on, this server is treated as an OAuth pass-through: the gateway proxies the upstream /.well-known/oauth-protected-resource metadata, emits spec-compliant 401 challenges when no bearer is supplied, and propagates upstream 401/403 responses. Only honored when Auth Type is None and 'Authorization' is in Extra Headers.">
                    <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                  </SimpleTooltip>
                </span>
                <p className="text-sm text-muted-foreground mt-1">
                  Forward upstream OAuth discovery and 401 challenges so clients negotiate OAuth directly with the
                  upstream MCP server.
                </p>
              </div>
              <MountedFormField
                name="oauth_passthrough"
                defaultValue={mcpServer?.oauth_passthrough ?? false}
                className="mb-0"
              >
                {(control) => <Switch aria-label="OAuth pass-through" {...switchControl(control)} />}
              </MountedFormField>
            </div>
          )}

          {showInternalDelegatePkceWarning && (
            <Alert variant="warning" className="mb-2">
              <TriangleAlert />
              <AlertTitle>Internal server with upstream OAuth delegation</AlertTitle>
              <AlertDescription>
                This MCP server is configured as internal-only but delegates auth to upstream. Anonymous users will be
                able to reach the upstream OAuth2 /authorize flow without a LiteLLM session. Ensure your upstream
                provider and network enforce access controls.
              </AlertDescription>
            </Alert>
          )}

          <MountedFormField
            label={
              <span className="text-sm font-medium text-foreground flex items-center">
                MCP Access Groups
                <SimpleTooltip content="Specify access groups for this MCP server. Users must be in at least one of these groups to access the server.">
                  <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                </SimpleTooltip>
              </span>
            }
            name="mcp_access_groups"
            className="mb-4"
          >
            {(control) => (
              <MultiSelect
                {...tagsControl(control)}
                options={availableAccessGroups.map((group) => ({ label: group, value: group }))}
                placeholder="Select existing groups or type to create new ones"
                className="rounded-lg"
              />
            )}
          </MountedFormField>

          <MountedFormField
            label={
              <span className="text-sm font-medium text-foreground flex items-center">
                Extra Headers
                <SimpleTooltip content="Forward custom headers from incoming requests to this MCP server (e.g., Authorization, X-Custom-Header, User-Agent)">
                  <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                </SimpleTooltip>
                {mcpServer?.extra_headers && mcpServer.extra_headers.length > 0 && (
                  <span className="ml-2 text-xs bg-info/15 text-info px-2 py-1 rounded-full">
                    {mcpServer.extra_headers.length} configured
                  </span>
                )}
              </span>
            }
            name="extra_headers"
          >
            {(control) => (
              <MultiSelect
                {...tagsControl(control)}
                placeholder={
                  mcpServer?.extra_headers && mcpServer.extra_headers.length > 0
                    ? `Currently: ${mcpServer.extra_headers.join(", ")}`
                    : "Enter header names (e.g., Authorization, X-Custom-Header)"
                }
                className="rounded-lg"
              />
            )}
          </MountedFormField>

          <Field>
            <FieldLabel>
              <span className="text-sm font-medium text-foreground flex items-center">
                Static Headers
                <SimpleTooltip content="Send these key-value headers with every request to this MCP server.">
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
