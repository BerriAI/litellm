import React, { useEffect } from "react";
import { Alert, Select, Tooltip, Collapse, Input, Space, Button, Switch } from "antd";
import { InfoCircleOutlined, MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { useFormContext } from "react-hook-form";
import { MCPServer, AUTH_TYPE } from "@/components/mcp_tools/types";
import {
  MountedFormField,
  bindControl,
  useMountedFieldArray,
  useMountedFormContext,
  useMountedWatch,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
const { Panel } = Collapse;

interface MCPPermissionManagementProps {
  availableAccessGroups: string[];
  mcpServer: MCPServer | null;
  searchValue: string;
  setSearchValue: (value: string) => void;
  getAccessGroupOptions: () => Array<{
    value: string;
    label: React.ReactNode;
  }>;
}

const MCPPermissionManagement: React.FC<MCPPermissionManagementProps> = ({
  availableAccessGroups,
  mcpServer,
  searchValue,
  setSearchValue,
  getAccessGroupOptions,
}) => {
  const form = useFormContext<MountedFormValues>();
  const { control } = useMountedFormContext();
  const staticHeaders = useMountedFieldArray(control, "static_headers");
  const watchedAuthType = useMountedWatch("auth_type");
  const isOAuth2 = watchedAuthType === AUTH_TYPE.OAUTH2;
  const isNoneAuth = watchedAuthType === AUTH_TYPE.NONE || watchedAuthType == null;
  const watchedExtraHeaders = useMountedWatch("extra_headers");
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
  const watchedDelegateAuth = useMountedWatch("delegate_auth_to_upstream");
  const watchedPublicInternet = useMountedWatch("available_on_public_internet");
  const showInternalDelegatePkceWarning = isOAuth2 && watchedDelegateAuth === true && watchedPublicInternet === false;

  // Set initial values when mcpServer changes
  useEffect(() => {
    if (mcpServer) {
      if (mcpServer.static_headers) {
        const headerRows = Object.entries(mcpServer.static_headers).map(([header, value]) => ({
          header,
          value: value != null ? String(value) : "",
        }));
        form.setValue("static_headers", headerRows);
      }
      if (Array.isArray(mcpServer.env_vars) && mcpServer.env_vars.length > 0) {
        form.setValue(
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
        form.setValue("allow_all_keys", mcpServer.allow_all_keys);
      }
      if (typeof mcpServer.available_on_public_internet === "boolean") {
        form.setValue("available_on_public_internet", mcpServer.available_on_public_internet);
      }
      if (typeof mcpServer.delegate_auth_to_upstream === "boolean") {
        form.setValue("delegate_auth_to_upstream", mcpServer.delegate_auth_to_upstream);
      }
      if (typeof mcpServer.oauth_passthrough === "boolean") {
        form.setValue("oauth_passthrough", mcpServer.oauth_passthrough);
      }
    } else {
      form.setValue("allow_all_keys", false);
      form.setValue("available_on_public_internet", true);
      form.setValue("delegate_auth_to_upstream", false);
      form.setValue("oauth_passthrough", false);
    }
  }, [mcpServer, form]);

  // delegate_auth_to_upstream is only honored server-side for oauth2 servers.
  // Force it back to false whenever the user switches away from oauth2 so a
  // stale toggle value doesn't get persisted unexpectedly.
  useEffect(() => {
    if (!isOAuth2) {
      form.setValue("delegate_auth_to_upstream", false);
    }
  }, [isOAuth2, form]);

  // oauth_passthrough is only honored for auth_type=none servers that forward
  // Authorization upstream. Force it back to false otherwise.
  useEffect(() => {
    if (!canEnableOAuthPassthrough) {
      form.setValue("oauth_passthrough", false);
    }
  }, [canEnableOAuthPassthrough, form]);

  return (
    <Collapse className="bg-gray-50 border border-gray-200 rounded-lg" expandIconPosition="end" ghost={false}>
      <Panel
        header={
          <div className="flex items-center">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <h3 className="text-lg font-semibold text-gray-900">Permission Management / Access Control</h3>
            </div>
            <p className="text-sm text-gray-600 ml-4">Configure access permissions and security settings (Optional)</p>
          </div>
        }
        key="permissions"
        className="border-0"
        forceRender
      >
        <div className="space-y-6 pt-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="text-sm font-medium text-gray-700 flex items-center">
                Allow All LiteLLM Keys
                <Tooltip title="When enabled, every API key can access this MCP server.">
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
              </span>
              <p className="text-sm text-gray-600 mt-1">
                Enable if this server should be &quot;public&quot; to all keys.
              </p>
            </div>
            <MountedFormField name="allow_all_keys" defaultValue={mcpServer?.allow_all_keys ?? false} className="mb-0">
              {(field) => <Switch id={field.id} checked={Boolean(field.value)} onChange={field.onChange} />}
            </MountedFormField>
          </div>

          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="text-sm font-medium text-gray-700 flex items-center">
                Internal network only
                <Tooltip title="When on, only requests from within your internal network are accepted. Turn off to allow external clients (other clusters, ChatGPT, etc). API key authentication is always required regardless of this setting.">
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
              </span>
              <p className="text-sm text-gray-600 mt-1">
                Turn on to restrict access to callers within your internal network only.
              </p>
            </div>
            <MountedFormField name="available_on_public_internet" defaultValue={true} className="mb-0">
              {(field) => (
                <Switch id={field.id} checked={!field.value} onChange={(checked) => field.onChange(!checked)} />
              )}
            </MountedFormField>
          </div>

          {isOAuth2 && (
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  Delegate auth to upstream (PKCE passthrough)
                  <Tooltip title="When on, LiteLLM skips its own API key/SSO check for this server and lets the client complete PKCE directly with the upstream MCP server. Only honored when Auth Type is oauth2. No spend tracking or per-key rate limiting will run on this route.">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
                <p className="text-sm text-gray-600 mt-1">
                  Bypass LiteLLM auth so clients authenticate directly with the upstream OAuth MCP server.
                </p>
              </div>
              <MountedFormField
                name="delegate_auth_to_upstream"
                defaultValue={mcpServer?.delegate_auth_to_upstream ?? false}
                className="mb-0"
              >
                {(field) => <Switch id={field.id} checked={Boolean(field.value)} onChange={field.onChange} />}
              </MountedFormField>
            </div>
          )}

          {canEnableOAuthPassthrough && (
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  OAuth pass-through
                  <Tooltip title="When on, this server is treated as an OAuth pass-through: the gateway proxies the upstream /.well-known/oauth-protected-resource metadata, emits spec-compliant 401 challenges when no bearer is supplied, and propagates upstream 401/403 responses. Only honored when Auth Type is None and 'Authorization' is in Extra Headers.">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
                <p className="text-sm text-gray-600 mt-1">
                  Forward upstream OAuth discovery and 401 challenges so clients negotiate OAuth directly with the
                  upstream MCP server.
                </p>
              </div>
              <MountedFormField
                name="oauth_passthrough"
                defaultValue={mcpServer?.oauth_passthrough ?? false}
                className="mb-0"
              >
                {(field) => <Switch id={field.id} checked={Boolean(field.value)} onChange={field.onChange} />}
              </MountedFormField>
            </div>
          )}

          {showInternalDelegatePkceWarning && (
            <Alert
              type="warning"
              showIcon
              className="mb-2"
              message="Internal server with upstream OAuth delegation"
              description="This MCP server is configured as internal-only but delegates auth to upstream. Anonymous users will be able to reach the upstream OAuth2 /authorize flow without a LiteLLM session. Ensure your upstream provider and network enforce access controls."
            />
          )}

          <MountedFormField
            label={
              <span className="text-sm font-medium text-gray-700 flex items-center">
                MCP Access Groups
                <Tooltip title="Specify access groups for this MCP server. Users must be in at least one of these groups to access the server.">
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
              </span>
            }
            name="mcp_access_groups"
            className="mb-4"
          >
            {(field) => (
              <Select
                {...bindControl<string[] | undefined>(field)}
                mode="tags"
                showSearch
                placeholder="Select existing groups or type to create new ones"
                optionFilterProp="value"
                filterOption={(input, option) => (option?.value ?? "").toLowerCase().includes(input.toLowerCase())}
                onSearch={(value) => setSearchValue(value)}
                tokenSeparators={[","]}
                options={getAccessGroupOptions()}
                maxTagCount="responsive"
                allowClear
              />
            )}
          </MountedFormField>

          <MountedFormField
            label={
              <span className="text-sm font-medium text-gray-700 flex items-center">
                Extra Headers
                <Tooltip title="Forward custom headers from incoming requests to this MCP server (e.g., Authorization, X-Custom-Header, User-Agent)">
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
                {mcpServer?.extra_headers && mcpServer.extra_headers.length > 0 && (
                  <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                    {mcpServer.extra_headers.length} configured
                  </span>
                )}
              </span>
            }
            name="extra_headers"
          >
            {(field) => (
              <Select
                {...bindControl<string[] | undefined>(field)}
                mode="tags"
                placeholder={
                  mcpServer?.extra_headers && mcpServer.extra_headers.length > 0
                    ? `Currently: ${mcpServer.extra_headers.join(", ")}`
                    : "Enter header names (e.g., Authorization, X-Custom-Header)"
                }
                className="rounded-lg"
                size="large"
                tokenSeparators={[","]}
                allowClear
              />
            )}
          </MountedFormField>

          <div>
            <div className="text-sm font-medium text-gray-700 flex items-center mb-2">
              Static Headers
              <Tooltip title="Send these key-value headers with every request to this MCP server.">
                <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
              </Tooltip>
            </div>
            <div className="space-y-3">
              {staticHeaders.fields.map((row, index) => (
                <Space key={row.id} className="flex w-full" align="baseline" size="middle">
                  <div className="flex-1">
                    <MountedFormField
                      name={`static_headers.${index}.header`}
                      required
                      rules={{ required: "Header name is required" }}
                    >
                      {(field) => (
                        <Input
                          {...bindControl<string | undefined>(field)}
                          size="large"
                          allowClear
                          className="rounded-lg"
                          placeholder="Header name (e.g., X-API-Key)"
                        />
                      )}
                    </MountedFormField>
                  </div>
                  <div className="flex-1">
                    <MountedFormField
                      name={`static_headers.${index}.value`}
                      required
                      rules={{ required: "Header value is required" }}
                    >
                      {(field) => (
                        <Input
                          {...bindControl<string | undefined>(field)}
                          size="large"
                          allowClear
                          className="rounded-lg"
                          placeholder="Header value"
                        />
                      )}
                    </MountedFormField>
                  </div>
                  <MinusCircleOutlined
                    onClick={() => staticHeaders.remove(index)}
                    className="text-gray-500 hover:text-red-500 cursor-pointer"
                  />
                </Space>
              ))}
              <Button type="dashed" onClick={() => staticHeaders.append({})} icon={<PlusOutlined />} block>
                Add Static Header
              </Button>
            </div>
          </div>
        </div>
      </Panel>
    </Collapse>
  );
};

export default MCPPermissionManagement;
