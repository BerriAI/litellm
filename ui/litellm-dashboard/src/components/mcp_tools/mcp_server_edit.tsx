import React, { useState, useEffect } from "react";
import { Form, Select, Button as AntdButton, Tooltip, Input } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { AUTH_TYPE, OAUTH_FLOW, MCPServer, MCPServerCostInfo } from "./types";
import { updateMCPServer, testMCPToolsListRequest } from "../networking";
import MCPServerCostConfig from "./mcp_server_cost_config";
import MCPPermissionManagement from "./MCPPermissionManagement";
import MCPToolConfiguration from "./mcp_tool_configuration";
import StdioConfiguration from "./StdioConfiguration";
import { validateMCPServerUrl, validateMCPServerName } from "./utils";
import NotificationsManager from "../molecules/notifications_manager";
import { useMcpOAuthFlow } from "@/hooks/useMcpOAuthFlow";

interface MCPServerEditProps {
  mcpServer: MCPServer;
  accessToken: string | null;
  onCancel: () => void;
  onSuccess: (server: MCPServer) => void;
  availableAccessGroups: string[];
}

const AUTH_TYPES_REQUIRING_AUTH_VALUE = [AUTH_TYPE.API_KEY, AUTH_TYPE.BEARER_TOKEN, AUTH_TYPE.BASIC];
const AUTH_TYPES_REQUIRING_CREDENTIALS = [...AUTH_TYPES_REQUIRING_AUTH_VALUE, AUTH_TYPE.OAUTH2];
const EDIT_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-edit-state";

const MCPServerEdit: React.FC<MCPServerEditProps> = ({
  mcpServer,
  accessToken,
  onCancel,
  onSuccess,
  availableAccessGroups,
}) => {
  const [form] = Form.useForm();
  const [costConfig, setCostConfig] = useState<MCPServerCostInfo>({});
  const [tools, setTools] = useState<any[]>([]);
  const [isLoadingTools, setIsLoadingTools] = useState(false);
  const [searchValue, setSearchValue] = useState<string>("");
  const [aliasManuallyEdited, setAliasManuallyEdited] = useState(false);
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [pendingRestoredValues, setPendingRestoredValues] = useState<Record<string, any> | null>(null);
  const authType = Form.useWatch("auth_type", form) as string | undefined;
  const transportType = Form.useWatch("transport", form) as string | undefined;
  const isStdioTransport = transportType === "stdio";
  const shouldShowAuthValueField = authType ? AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(authType) : false;
  const isOAuthAuthType = authType === AUTH_TYPE.OAUTH2;
  const oauthFlowTypeValue = Form.useWatch("oauth_flow_type", form) as string | undefined;
  const isM2MFlow = isOAuthAuthType && oauthFlowTypeValue === OAUTH_FLOW.M2M;

  const [oauthAccessToken, setOauthAccessToken] = useState<string | null>(null);

  const persistEditUiState = () => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const values = form.getFieldsValue(true);
      window.sessionStorage.setItem(
        EDIT_OAUTH_UI_STATE_KEY,
        JSON.stringify({
          serverId: mcpServer.server_id,
          formValues: values,
          costConfig,
          allowedTools,
          searchValue,
          aliasManuallyEdited,
        }),
      );
    } catch (err) {
      console.warn("Failed to persist MCP edit state", err);
    }
  };

  const {
    startOAuthFlow,
    status: oauthStatus,
    error: oauthError,
    tokenResponse: oauthTokenResponse,
  } = useMcpOAuthFlow({
    accessToken,
    getCredentials: () => form.getFieldValue("credentials"),
    getTemporaryPayload: () => {
      const values = form.getFieldsValue(true);
      const url = values.url || mcpServer.url;
      const transport = values.transport || mcpServer.transport;
      if (!url || !transport) {
        return null;
      }
      const staticHeaders = Array.isArray(values.static_headers)
        ? values.static_headers.reduce((acc: Record<string, string>, entry: Record<string, string>) => {
            const header = entry?.header?.trim();
            if (!header) {
              return acc;
            }
            acc[header] = entry?.value ?? "";
            return acc;
          }, {})
        : ({} as Record<string, string>);

      return {
        server_id: mcpServer.server_id,
        server_name: values.server_name || mcpServer.server_name || mcpServer.alias,
        alias: values.alias || mcpServer.alias,
        description: values.description || mcpServer.description,
        url,
        transport,
        auth_type: AUTH_TYPE.OAUTH2,
        credentials: values.credentials,
        mcp_access_groups: values.mcp_access_groups || mcpServer.mcp_access_groups,
        static_headers: staticHeaders,
        command: values.command,
        args: values.args,
        env: values.env,
      };
    },
    onTokenReceived: (token) => {
      setOauthAccessToken(token?.access_token ?? null);
    },
    onBeforeRedirect: persistEditUiState,
  });

  const initialStaticHeaders = React.useMemo(() => {
    if (!mcpServer.static_headers) {
      return [];
    }
    return Object.entries(mcpServer.static_headers).map(([header, value]) => ({
      header,
      value: value != null ? String(value) : "",
    }));
  }, [mcpServer.static_headers]);

  const initialEnvJson = React.useMemo(() => {
    const env = mcpServer.env ?? undefined;
    if (!env || Object.keys(env).length === 0) {
      return "";
    }
    try {
      return JSON.stringify(env, null, 2);
    } catch {
      return "";
    }
  }, [mcpServer.env]);


  const initialValues = React.useMemo(
    () => ({
      ...mcpServer,
      static_headers: initialStaticHeaders,
      oauth_flow_type: mcpServer.token_url ? OAUTH_FLOW.M2M : OAUTH_FLOW.INTERACTIVE,
    }),
    [mcpServer, initialStaticHeaders, initialEnvJson],
  );

  // Initialize cost config from existing server data
  useEffect(() => {
    if (mcpServer.mcp_info?.mcp_server_cost_info) {
      setCostConfig(mcpServer.mcp_info.mcp_server_cost_info);
    }
  }, [mcpServer]);

  // Initialize allowed tools from existing server data
  useEffect(() => {
    if (mcpServer.allowed_tools) {
      setAllowedTools(mcpServer.allowed_tools);
    }
  }, [mcpServer]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const storedState = window.sessionStorage.getItem(EDIT_OAUTH_UI_STATE_KEY);
    if (!storedState) {
      return;
    }

    try {
      const parsed = JSON.parse(storedState);
      if (!parsed || parsed.serverId !== mcpServer.server_id) {
        return;
      }
      if (parsed.formValues) {
        setPendingRestoredValues({ ...mcpServer, ...parsed.formValues });
      }
      if (parsed.costConfig) {
        setCostConfig(parsed.costConfig);
      }
      if (parsed.allowedTools) {
        setAllowedTools(parsed.allowedTools);
      }
      if (parsed.searchValue) {
        setSearchValue(parsed.searchValue);
      }
      if (typeof parsed.aliasManuallyEdited === "boolean") {
        setAliasManuallyEdited(parsed.aliasManuallyEdited);
      }
    } catch (err) {
      console.error("Failed to restore MCP edit state", err);
    } finally {
      window.sessionStorage.removeItem(EDIT_OAUTH_UI_STATE_KEY);
    }
  }, [form, mcpServer]);

  useEffect(() => {
    if (!pendingRestoredValues) {
      return;
    }
    const transport = pendingRestoredValues.transport || mcpServer.transport;
    if (transport && transport !== form.getFieldValue("transport")) {
      form.setFieldsValue({ transport });
      return;
    }
    form.setFieldsValue(pendingRestoredValues);
    setPendingRestoredValues(null);
  }, [pendingRestoredValues, form, mcpServer.transport]);

  // Transform string array to object array for initial form values
  useEffect(() => {
    if (mcpServer.mcp_access_groups) {
      // If access groups are objects, extract the name property; if strings, use as is
      const groupNames = mcpServer.mcp_access_groups.map((g: any) => (typeof g === "string" ? g : g.name || String(g)));
      form.setFieldValue("mcp_access_groups", groupNames);
    }
  }, [mcpServer]);

  // Fetch tools when component mounts
  useEffect(() => {
    fetchTools();
  }, [mcpServer, accessToken, oauthAccessToken]);

  const fetchTools = async () => {
    if (!accessToken) return;

    // HTTP/SSE requires a URL; stdio does not.
    if (mcpServer.transport !== "stdio" && !mcpServer.url) return;

    const isM2M = mcpServer.auth_type === AUTH_TYPE.OAUTH2 && !!mcpServer.token_url;
    if (mcpServer.auth_type === AUTH_TYPE.OAUTH2 && !isM2M && !oauthAccessToken) {
      return;
    }

    setIsLoadingTools(true);

    try {
      // Prepare the MCP server config from existing server data
      const mcpServerConfig = {
        server_id: mcpServer.server_id,
        server_name: mcpServer.server_name,
        url: mcpServer.url,
        transport: mcpServer.transport,
        auth_type: mcpServer.auth_type,
        mcp_info: mcpServer.mcp_info,
        authorization_url: mcpServer.authorization_url,
        token_url: mcpServer.token_url,
        registration_url: mcpServer.registration_url,
        command: mcpServer.command,
        args: mcpServer.args,
        env: mcpServer.env,
      };

      const toolsResponse = await testMCPToolsListRequest(accessToken, mcpServerConfig, oauthAccessToken);

      if (toolsResponse.tools && !toolsResponse.error) {
        setTools(toolsResponse.tools);
      } else {
        console.error("Failed to fetch tools:", toolsResponse.message);
        setTools([]);
      }
    } catch (error) {
      console.error("Tools fetch error:", error);
      setTools([]);
    } finally {
      setIsLoadingTools(false);
    }
  };

  // Generate options with existing groups and potential new group
  const getAccessGroupOptions = () => {
    const existingOptions = availableAccessGroups.map((group: string) => ({
      value: group,
      label: (
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span className="font-medium">{group}</span>
        </div>
      ),
    }));

    // If search value doesn't match any existing group and is not empty, add "create new group" option
    if (
      searchValue &&
      !availableAccessGroups.some((group) => group.toLowerCase().includes(searchValue.toLowerCase()))
    ) {
      existingOptions.push({
        value: searchValue,
        label: (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span className="font-medium">{searchValue}</span>
            <span className="text-gray-400 text-xs ml-1">create new group</span>
          </div>
        ),
      });
    }

    return existingOptions;
  };

  const handleTransportChange = (value: string) => {
    // Clear fields that are not relevant for the selected transport.
    if (value === "stdio") {
      form.setFieldsValue({
        url: undefined,
        auth_type: undefined,
        credentials: undefined,
        authorization_url: undefined,
        token_url: undefined,
        registration_url: undefined,
      });
    } else {
      form.setFieldsValue({
        command: undefined,
        args: undefined,
        env_json: undefined,
        stdio_config: undefined,
      });
    }
  };

  const handleSave = async (values: Record<string, any>) => {
    if (!accessToken) return;
    try {
      // Ensure access groups is always a string array
      const {
        static_headers: staticHeadersList,
        credentials: credentialValues,
        stdio_config: rawStdioConfig,
        env_json: rawEnvJson,
        command: rawCommand,
        args: rawArgs,
        allow_all_keys: allowAllKeysRaw,
        available_on_public_internet: availableOnPublicInternetRaw,
        ...restValues
      } = values;

      const accessGroups = (restValues.mcp_access_groups || []).map((g: any) =>
        typeof g === "string" ? g : g.name || String(g),
      );

      const staticHeaders = Array.isArray(staticHeadersList)
        ? staticHeadersList.reduce((acc: Record<string, string>, entry: Record<string, string>) => {
            const header = entry?.header?.trim();
            if (!header) {
              return acc;
            }
            acc[header] = entry?.value ?? "";
            return acc;
          }, {})
        : ({} as Record<string, string>);

      const credentialsPayload =
        credentialValues && typeof credentialValues === "object"
          ? Object.entries(credentialValues).reduce((acc: Record<string, any>, [key, value]) => {
              if (value === undefined || value === null || value === "") {
                return acc;
              }
              if (key === "scopes") {
                if (Array.isArray(value)) {
                  const filteredScopes = value.filter((scope) => scope != null && scope !== "");
                  if (filteredScopes.length > 0) {
                    acc[key] = filteredScopes;
                  }
                }
              } else {
                acc[key] = value;
              }
              return acc;
            }, {})
          : undefined;

      let stdioFields: Record<string, any> = {};

      if (restValues.transport === "stdio") {
        // Prefer JSON config if provided (matches Create screen behavior)
        if (rawStdioConfig) {
          try {
            const stdioConfig = JSON.parse(rawStdioConfig);

            let actualConfig = stdioConfig;
            if (stdioConfig?.mcpServers && typeof stdioConfig.mcpServers === "object") {
              const serverNames = Object.keys(stdioConfig.mcpServers);
              if (serverNames.length > 0) {
                actualConfig = stdioConfig.mcpServers[serverNames[0]];
              }
            }

            const parsedArgs = Array.isArray(actualConfig?.args)
              ? actualConfig.args.map((v: any) => String(v)).filter((v: string) => v.trim() !== "")
              : [];

            const parsedEnv =
              actualConfig?.env && typeof actualConfig.env === "object" && !Array.isArray(actualConfig.env)
                ? Object.entries(actualConfig.env).reduce((acc: Record<string, string>, [k, v]) => {
                    if (k == null || String(k).trim() === "") return acc;
                    acc[String(k)] = v == null ? "" : String(v);
                    return acc;
                  }, {})
                : {};

            stdioFields = {
              command: actualConfig?.command ? String(actualConfig.command) : undefined,
              args: parsedArgs,
              env: parsedEnv,
            };

            if (!stdioFields.command) {
              NotificationsManager.fromBackend("Stdio configuration must include a command");
              return;
            }
          } catch {
            NotificationsManager.fromBackend("Invalid JSON in stdio configuration");
            return;
          }
        } else {
          // Dedicated fields path (command/args + env JSON)
          let parsedEnv: Record<string, string> = {};
          if (rawEnvJson) {
            try {
              const env = JSON.parse(rawEnvJson);
              if (env && typeof env === "object" && !Array.isArray(env)) {
                parsedEnv = Object.entries(env).reduce((acc: Record<string, string>, [k, v]) => {
                  if (k == null || String(k).trim() === "") return acc;
                  acc[String(k)] = v == null ? "" : String(v);
                  return acc;
                }, {});
              }
            } catch {
              NotificationsManager.fromBackend("Invalid JSON in stdio env configuration");
              return;
            }
          }
          const parsedArgs = Array.isArray(rawArgs)
            ? rawArgs.map((v: any) => String(v)).filter((v: string) => v.trim() !== "")
            : [];

          const parsedCommand = rawCommand ? String(rawCommand).trim() : "";
          if (!parsedCommand) {
            NotificationsManager.fromBackend("Stdio transport requires a command");
            return;
          }

          stdioFields = {
            command: parsedCommand,
            args: parsedArgs,
            env: parsedEnv,
          };
        }
      }

      // Prepare the payload with cost configuration and permission fields
      const mcpInfoServerName =
        restValues.server_name ||
        restValues.url ||
        mcpServer.server_name ||
        mcpServer.url ||
        restValues.alias ||
        mcpServer.alias ||
        "unknown";

      const payload: Record<string, any> = {
        ...restValues,
        ...stdioFields,
        // Remove UI-only fields
        stdio_config: undefined,
        env_json: undefined,
        server_id: mcpServer.server_id,
        mcp_info: {
          server_name: mcpInfoServerName,
          description: restValues.description,
          mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null,
        },
        mcp_access_groups: accessGroups,
        alias: restValues.alias,
        // Include permission management fields
        extra_headers: restValues.extra_headers || [],
        allowed_tools: allowedTools.length > 0 ? allowedTools : null,
        disallowed_tools: restValues.disallowed_tools || [],
        static_headers: staticHeaders,
        allow_all_keys: Boolean(allowAllKeysRaw ?? mcpServer.allow_all_keys),
        available_on_public_internet: Boolean(availableOnPublicInternetRaw ?? mcpServer.available_on_public_internet),
      };

      const includeCredentials = restValues.auth_type && AUTH_TYPES_REQUIRING_CREDENTIALS.includes(restValues.auth_type);

      if (includeCredentials && credentialsPayload && Object.keys(credentialsPayload).length > 0) {
        payload.credentials = credentialsPayload;
      }

      const updated = await updateMCPServer(accessToken, payload);
      NotificationsManager.success("MCP Server updated successfully");
      onSuccess(updated);
    } catch (error: any) {
      NotificationsManager.fromBackend("Failed to update MCP Server" + (error?.message ? `: ${error.message}` : ""));
    }
  };

  return (
    <TabGroup>
      <TabList className="grid w-full grid-cols-2">
        <Tab>Server Configuration</Tab>
        <Tab>Cost Configuration</Tab>
      </TabList>
      <TabPanels className="mt-6">
        <TabPanel>
          <Form form={form} onFinish={handleSave} initialValues={initialValues} layout="vertical">
            <Form.Item
              label="MCP Server Name"
              name="server_name"
              rules={[
                {
                  validator: (_, value) => validateMCPServerName(value),
                },
              ]}
            >
              <Input className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500" />
            </Form.Item>
            <Form.Item
              label="Alias"
              name="alias"
              rules={[
                {
                  validator: (_, value) => validateMCPServerName(value),
                },
              ]}
            >
              <Input
                onChange={() => setAliasManuallyEdited(true)}
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>
            <Form.Item label="Description" name="description">
              <Input className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500" />
            </Form.Item>
            <Form.Item label="Transport Type" name="transport" rules={[{ required: true }]}>
              <Select onChange={handleTransportChange}>
                <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                <Select.Option value="http">Streamable HTTP (Recommended)</Select.Option>
                <Select.Option value="stdio">Standard Input/Output (stdio)</Select.Option>
              </Select>
            </Form.Item>

            {/* URL/Auth fields are only applicable for HTTP/SSE */}
            {!isStdioTransport && (
              <Form.Item
                label="MCP Server URL"
                name="url"
                rules={[
                  { required: true, message: "Please enter a server URL" },
                  { validator: (_, value) => validateMCPServerUrl(value) },
                ]}
              >
                <Input
                  placeholder="https://your-mcp-server.com"
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>
            )}

            {!isStdioTransport && (
              <Form.Item label="Authentication" name="auth_type" rules={[{ required: true }]}>
                <Select>
                  <Select.Option value="none">None</Select.Option>
                  <Select.Option value="api_key">API Key</Select.Option>
                  <Select.Option value="bearer_token">Bearer Token</Select.Option>
                  <Select.Option value="basic">Basic Auth</Select.Option>
                  <Select.Option value="oauth2">OAuth</Select.Option>
                </Select>
              </Form.Item>
            )}

            {isStdioTransport && (
              <div className="rounded-lg border border-gray-200 p-4 space-y-4">
                <p className="text-sm text-gray-600">
                  Configure the stdio transport used to launch the MCP server process. You can either fill in the fields
                  below or paste a JSON configuration.
                </p>

                <Form.Item
                  label="Command"
                  name="command"
                  rules={[{ required: true, message: "Please enter a command for stdio transport" }]}
                >
                  <Input
                    placeholder="e.g., npx"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>

                <Form.Item
                  label="Args"
                  name="args"
                >
                  <Select
                    mode="tags"
                    size="large"
                    tokenSeparators={[","]}
                    placeholder="Add args (press enter or comma)"
                    className="rounded-lg"
                  />
                </Form.Item>

                <Form.Item
                  label="Environment (JSON object)"
                  name="env_json"
                  rules={[
                    {
                      validator: (_, value) => {
                        if (!value) return Promise.resolve();
                        try {
                          const parsed = JSON.parse(value);
                          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                            return Promise.resolve();
                          }
                          return Promise.reject(new Error("Env must be a JSON object"));
                        } catch {
                          return Promise.reject(new Error("Please enter valid JSON"));
                        }
                      },
                    },
                  ]}
                >
                  <Input.TextArea
                    rows={6}
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 font-mono text-sm"
                    placeholder={`{\n  \"KEY\": \"value\"\n}`}
                  />
                </Form.Item>

                {/* Optional JSON config (if provided, it overrides command/args/env on save) */}
                <StdioConfiguration isVisible={true} required={false} />
              </div>
            )}

            {!isStdioTransport && shouldShowAuthValueField && (
              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    Authentication Value
                    <Tooltip title="Token, password, or header value to send with each request for the selected auth type.">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name={["credentials", "auth_value"]}
                rules={[
                  {
                    validator: (_, value) =>
                      value && typeof value === "string" && value.trim() === ""
                        ? Promise.reject(new Error("Authentication value cannot be empty"))
                        : Promise.resolve(),
                  },
                ]}
              >
                <Input.Password
                  placeholder="Enter token or secret (leave blank to keep existing)"
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>
            )}

            {!isStdioTransport && isOAuthAuthType && (
              <>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      OAuth Client ID (optional)
                      <Tooltip title="Provide only if your MCP server cannot handle dynamic client registration.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "client_id"]}
                >
                  <Input.Password
                    placeholder="Enter OAuth client ID (leave blank to keep existing)"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      OAuth Client Secret (optional)
                      <Tooltip title="Provide only if your MCP server cannot handle dynamic client registration.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "client_secret"]}
                >
                  <Input.Password
                    placeholder="Enter OAuth client secret (leave blank to keep existing)"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      OAuth Scopes (optional)
                      <Tooltip title="Add scopes to override the default scope list used for this MCP server.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "scopes"]}
                >
                  <Select
                    mode="tags"
                    tokenSeparators={[","]}
                    placeholder="Add scopes"
                    className="rounded-lg"
                    size="large"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      Authorization URL Override (optional)
                      <Tooltip title="Optional override for the authorization endpoint.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name="authorization_url"
                >
                  <Input
                    placeholder="https://example.com/oauth/authorize"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      Token URL Override (optional)
                      <Tooltip title="Optional override for the token endpoint.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name="token_url"
                >
                  <Input
                    placeholder="https://example.com/oauth/token"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      Registration URL Override (optional)
                      <Tooltip title="Optional override for the dynamic client registration endpoint.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name="registration_url"
                >
                  <Input
                    placeholder="https://example.com/oauth/register"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <div className="rounded-lg border border-dashed border-gray-300 p-4 space-y-2">
                  <p className="text-sm text-gray-600">Use OAuth to fetch a fresh access token and temporarily save it in the session as the authentication value.</p>
                  <Button
                    variant="secondary"
                    onClick={startOAuthFlow}
                    disabled={oauthStatus === "authorizing" || oauthStatus === "exchanging"}
                  >
                    {oauthStatus === "authorizing"
                      ? "Waiting for authorization..."
                      : oauthStatus === "exchanging"
                        ? "Exchanging authorization code..."
                        : "Authorize & Fetch Token"}
                  </Button>
                  {oauthError && <p className="text-sm text-red-500">{oauthError}</p>}
                  {oauthStatus === "success" && oauthTokenResponse?.access_token && (
                    <p className="text-sm text-green-600">
                      Token fetched. Expires in {oauthTokenResponse.expires_in ?? "?"} seconds.
                    </p>
                  )}
                </div>
              </>
            )}

            {/* Permission Management / Access Control Section */}
            <div className="mt-6">
              <MCPPermissionManagement
                availableAccessGroups={availableAccessGroups}
                mcpServer={mcpServer}
                searchValue={searchValue}
                setSearchValue={setSearchValue}
                getAccessGroupOptions={getAccessGroupOptions}
              />
            </div>

            {/* Tool Configuration Section */}
            <div className="mt-6">
              <MCPToolConfiguration
                accessToken={accessToken}
                oauthAccessToken={oauthAccessToken}
                formValues={{
                  server_id: mcpServer.server_id,
                  server_name: mcpServer.server_name,
                  url: mcpServer.url,
                  transport: mcpServer.transport,
                  auth_type: mcpServer.auth_type,
                  mcp_info: mcpServer.mcp_info,
                  oauth_flow_type: mcpServer.token_url ? OAUTH_FLOW.M2M : OAUTH_FLOW.INTERACTIVE,
                }}
                allowedTools={allowedTools}
                existingAllowedTools={mcpServer.allowed_tools || null}
                onAllowedToolsChange={setAllowedTools}
              />
            </div>

            <div className="flex justify-end gap-2">
              <AntdButton onClick={onCancel}>Cancel</AntdButton>
              <Button type="submit">Save Changes</Button>
            </div>
          </Form>
        </TabPanel>

        <TabPanel>
          <div className="space-y-6">
            <MCPServerCostConfig value={costConfig} onChange={setCostConfig} tools={tools} disabled={isLoadingTools} />

            <div className="flex justify-end gap-2">
              <AntdButton onClick={onCancel}>Cancel</AntdButton>
              <Button onClick={() => form.submit()}>Save Changes</Button>
            </div>
          </div>
        </TabPanel>
      </TabPanels>
    </TabGroup>
  );
};

export default MCPServerEdit;
