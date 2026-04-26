import React, { useState, useEffect } from "react";
import { Form, Select, Button as AntdButton, Tooltip, Input, InputNumber } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { AUTH_TYPE, OAUTH_FLOW, MCPServer, MCPServerCostInfo, TRANSPORT } from "./types";
import { updateMCPServer, listMCPTools } from "../networking";
import MCPServerCostConfig from "./mcp_server_cost_config";
import MCPPermissionManagement from "./MCPPermissionManagement";
import MCPToolConfiguration from "./mcp_tool_configuration";
import StdioConfiguration from "./StdioConfiguration";
import MCPLogoSelector from "./MCPLogoSelector";
import { validateMCPServerUrl, validateMCPServerName } from "./utils";
import NotificationsManager from "../molecules/notifications_manager";
import { useMcpOAuthFlow } from "@/hooks/useMcpOAuthFlow";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";

interface MCPServerEditProps {
  mcpServer: MCPServer;
  accessToken: string | null;
  onCancel: () => void;
  onSuccess: (server: MCPServer) => void;
  availableAccessGroups: string[];
}

const AUTH_TYPES_REQUIRING_AUTH_VALUE = [AUTH_TYPE.API_KEY, AUTH_TYPE.BEARER_TOKEN, AUTH_TYPE.TOKEN, AUTH_TYPE.BASIC];
const AUTH_TYPES_REQUIRING_CREDENTIALS = [...AUTH_TYPES_REQUIRING_AUTH_VALUE, AUTH_TYPE.OAUTH2, AUTH_TYPE.AWS_SIGV4];
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
  const [toolsError, setToolsError] = useState<string | null>(null);
  const [searchValue, setSearchValue] = useState<string>("");
  const [aliasManuallyEdited, setAliasManuallyEdited] = useState(false);
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [toolNameToDisplayName, setToolNameToDisplayName] = useState<Record<string, string>>({});
  const [toolNameToDescription, setToolNameToDescription] = useState<Record<string, string>>({});
  const [pendingRestoredValues, setPendingRestoredValues] = useState<Record<string, any> | null>(null);
  const [logoUrl, setLogoUrl] = useState<string | undefined>(mcpServer.mcp_info?.logo_url || undefined);
  const authType = Form.useWatch("auth_type", form) as string | undefined;
  const transportType = Form.useWatch("transport", form) as string | undefined;
  const isStdioTransport = transportType === "stdio";
  const isOpenAPITransport = transportType === TRANSPORT.OPENAPI;
  const isMCPTransport = !isStdioTransport && !isOpenAPITransport;
  const shouldShowAuthValueField = authType ? AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(authType) : false;
  const isOAuthAuthType = authType === AUTH_TYPE.OAUTH2;
  const isAwsSigV4AuthType = authType === AUTH_TYPE.AWS_SIGV4;
  const oauthFlowTypeValue = Form.useWatch("oauth_flow_type", form) as string | undefined;
  const isM2MFlow = isOAuthAuthType && oauthFlowTypeValue === OAUTH_FLOW.M2M;

  const [oauthAccessToken, setOauthAccessToken] = useState<string | null>(null);

  // Watch form fields that affect tool fetching
  const currentUrl = Form.useWatch("url", form);
  const currentSpecPath = Form.useWatch("spec_path", form);
  const currentServerName = Form.useWatch("server_name", form);
  const currentAuthType = Form.useWatch("auth_type", form);
  const currentStaticHeaders = Form.useWatch("static_headers", form);
  const currentCredentials = Form.useWatch("credentials", form);
  const currentAuthorizationUrl = Form.useWatch("authorization_url", form);
  const currentTokenUrl = Form.useWatch("token_url", form);
  const currentRegistrationUrl = Form.useWatch("registration_url", form);

  const persistEditUiState = () => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const values = form.getFieldsValue(true);
      setSecureItem(
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
      
      if (token?.access_token) {
        const credentials = {
          access_token: token.access_token,
          ...(token.refresh_token && { refresh_token: token.refresh_token }),
          ...(token.expires_in && { expires_in: token.expires_in }),
          ...(token.scope && { scope: token.scope }),
        };
        
        form.setFieldsValue({ credentials });
        
        NotificationsManager.success(
          "OAuth authorization successful! Please click 'Update MCP Server' to save the credentials."
        );
      }
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


  // If server has spec_path, show it as "openapi" transport in the UI
  const effectiveTransport = React.useMemo(() => {
    if (mcpServer.spec_path && mcpServer.transport !== "stdio") {
      return TRANSPORT.OPENAPI;
    }
    return mcpServer.transport;
  }, [mcpServer]);

  const initialValues = React.useMemo(
    () => ({
      ...mcpServer,
      transport: effectiveTransport,
      static_headers: initialStaticHeaders,
      extra_headers: mcpServer.extra_headers || [],
      oauth_flow_type: mcpServer.token_url ? OAUTH_FLOW.M2M : OAUTH_FLOW.INTERACTIVE,
      token_validation_json: mcpServer.token_validation
        ? JSON.stringify(mcpServer.token_validation, null, 2)
        : undefined,
    }),
    [mcpServer, effectiveTransport, initialStaticHeaders, initialEnvJson],
  );

  // Initialize cost config from existing server data
  useEffect(() => {
    if (mcpServer.mcp_info?.mcp_server_cost_info) {
      setCostConfig(mcpServer.mcp_info.mcp_server_cost_info);
    }
  }, [mcpServer]);

  // Initialize allowed tools and tool overrides from existing server data
  useEffect(() => {
    if (mcpServer.allowed_tools) {
      setAllowedTools(mcpServer.allowed_tools);
    }
    setToolNameToDisplayName(mcpServer.tool_name_to_display_name ?? {});
    setToolNameToDescription(mcpServer.tool_name_to_description ?? {});
  }, [mcpServer]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const storedState = getSecureItem(EDIT_OAUTH_UI_STATE_KEY);
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

  // Fetch tools when component mounts for a saved server
  useEffect(() => {
    if (!mcpServer.server_id || mcpServer.server_id.trim() === "") {
      return;
    }
    fetchTools();
  }, [mcpServer, accessToken]);

  const fetchTools = async () => {
    if (!accessToken || !mcpServer.server_id) return;

    setIsLoadingTools(true);
    setToolsError(null);

    try {
      // Use the GET endpoint which looks up stored credentials by server_id,
      // rather than POST /test/tools/list which requires inline credentials.
      const toolsResponse = await listMCPTools(accessToken, mcpServer.server_id);

      if (toolsResponse.tools && !toolsResponse.error) {
        setTools(toolsResponse.tools);
      } else {
        console.error("Failed to fetch tools:", toolsResponse.message);
        setTools([]);
        setToolsError(toolsResponse.message || "Failed to load tools");
      }
    } catch (error) {
      console.error("Tools fetch error:", error);
      setTools([]);
      setToolsError(error instanceof Error ? error.message : "Failed to load tools");
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
        spec_path: undefined,
        auth_type: undefined,
        credentials: undefined,
        authorization_url: undefined,
        token_url: undefined,
        registration_url: undefined,
      });
    } else if (value === TRANSPORT.OPENAPI) {
      form.setFieldsValue({
        url: undefined,
        command: undefined,
        args: undefined,
        env_json: undefined,
        stdio_config: undefined,
      });
    } else {
      form.setFieldsValue({
        spec_path: undefined,
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
        token_validation_json: rawTokenValidationJson,
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

      // Map "openapi" transport to "http" for the backend
      if (restValues.transport === TRANSPORT.OPENAPI) {
        restValues.transport = "http";
      }

      // Parse token_validation JSON if provided
      let tokenValidation: Record<string, any> | null = null;
      if (rawTokenValidationJson && rawTokenValidationJson.trim() !== "") {
        try {
          tokenValidation = JSON.parse(rawTokenValidationJson);
        } catch {
          NotificationsManager.fromBackend("Invalid JSON in Token Validation Rules");
          return;
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
          logo_url: logoUrl || undefined,
          mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null,
        },
        mcp_access_groups: accessGroups,
        alias: restValues.alias,
        // Include permission management fields
        extra_headers: restValues.extra_headers || [],
        allowed_tools: allowedTools.length > 0 ? allowedTools : null,
        tool_name_to_display_name: Object.keys(toolNameToDisplayName).length > 0 ? toolNameToDisplayName : null,
        tool_name_to_description: Object.keys(toolNameToDescription).length > 0 ? toolNameToDescription : null,
        disallowed_tools: restValues.disallowed_tools || [],
        static_headers: staticHeaders,
        allow_all_keys: Boolean(allowAllKeysRaw ?? mcpServer.allow_all_keys),
        available_on_public_internet: Boolean(availableOnPublicInternetRaw ?? mcpServer.available_on_public_internet),
        // Include token_validation when it is set (non-null) or when clearing an existing value
        ...(tokenValidation !== null || mcpServer.token_validation
          ? { token_validation: tokenValidation }
          : {}),
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
            <MCPLogoSelector value={logoUrl} onChange={setLogoUrl} />
            <Form.Item label="Transport Type" name="transport" rules={[{ required: true }]}>
              <Select onChange={handleTransportChange}>
                <Select.Option value="http">Streamable HTTP (Recommended)</Select.Option>
                <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                <Select.Option value="stdio">Standard Input/Output (stdio)</Select.Option>
                <Select.Option value={TRANSPORT.OPENAPI}>OpenAPI Spec</Select.Option>
              </Select>
            </Form.Item>

            {/* URL field - only for HTTP/SSE */}
            {isMCPTransport && (
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

            {/* OpenAPI Spec URL - only for OpenAPI transport */}
            {isOpenAPITransport && (
              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    OpenAPI Spec URL
                    <Tooltip title="URL to an OpenAPI specification (JSON or YAML). MCP tools will be automatically generated from the API endpoints defined in the spec.">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="spec_path"
                rules={[{ required: true, message: "Please enter an OpenAPI spec URL" }]}
              >
                <Input
                  placeholder="https://petstore3.swagger.io/api/v3/openapi.json"
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>
            )}

            {/* Authentication - for HTTP, SSE, and OpenAPI */}
            {!isStdioTransport && (
              <Form.Item label="Authentication" name="auth_type" rules={[{ required: true }]}>
                <Select>
                  <Select.Option value="none">None</Select.Option>
                  <Select.Option value="api_key">API Key</Select.Option>
                  <Select.Option value="bearer_token">Bearer Token</Select.Option>
                  <Select.Option value="token">Token</Select.Option>
                  <Select.Option value="basic">Basic Auth</Select.Option>
                  <Select.Option value="oauth2">OAuth</Select.Option>
                  <Select.Option value="aws_sigv4">AWS SigV4 (Bedrock AgentCore MCPs)</Select.Option>
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
                {!isM2MFlow && (
                  <>
                    <Form.Item
                      label={
                        <span className="text-sm font-medium text-gray-700 flex items-center">
                          Token Validation Rules (optional)
                          <Tooltip title='JSON object of key-value rules checked against the OAuth token response before storing. Supports dot-notation for nested fields (e.g. {"organization": "my-org", "team.id": "123"}). Tokens that fail validation are rejected with HTTP 403.'>
                            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                          </Tooltip>
                        </span>
                      }
                      name="token_validation_json"
                      rules={[
                        {
                          validator: (_: any, value: string) => {
                            if (!value || value.trim() === "") return Promise.resolve();
                            try {
                              JSON.parse(value);
                              return Promise.resolve();
                            } catch {
                              return Promise.reject(new Error("Must be valid JSON"));
                            }
                          },
                        },
                      ]}
                    >
                      <Input.TextArea
                        placeholder={'{\n  "organization": "my-org",\n  "team.id": "123"\n}'}
                        rows={4}
                        className="font-mono text-sm rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                      />
                    </Form.Item>
                    <Form.Item
                      label={
                        <span className="text-sm font-medium text-gray-700 flex items-center">
                          Token Storage TTL (seconds, optional)
                          <Tooltip title="How long to cache each user's OAuth access token in Redis before evicting it (regardless of the token's own expires_in). Leave blank to derive the TTL from the token's expires_in, or fall back to the 12-hour default.">
                            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                          </Tooltip>
                        </span>
                      }
                      name="token_storage_ttl_seconds"
                    >
                      <InputNumber
                        min={1}
                        placeholder="e.g. 3600"
                        style={{ width: "100%" }}
                        className="rounded-lg"
                      />
                    </Form.Item>
                  </>
                )}
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

            {!isStdioTransport && isAwsSigV4AuthType && (
              <>
                <p className="text-sm text-gray-500 mb-2">
                  For MCP servers hosted on AWS Bedrock AgentCore.{" "}
                  <a href="https://docs.litellm.ai/docs/mcp_aws_sigv4" target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-700">
                    View docs &rarr;
                  </a>
                </p>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      AWS Region
                      <Tooltip title="AWS region for SigV4 signing (e.g., us-east-1)">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "aws_region_name"]}
                  rules={[]}
                >
                  <Input
                    placeholder="us-east-1 (leave blank to keep existing)"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      AWS Service Name
                      <Tooltip title="AWS service name for SigV4 signing. Defaults to 'bedrock-agentcore'.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "aws_service_name"]}
                >
                  <Input
                    placeholder="bedrock-agentcore (leave blank to keep existing)"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      AWS Access Key ID
                      <Tooltip title="Optional. If not provided, falls back to the boto3 credential chain (IAM role, env vars, etc.).">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "aws_access_key_id"]}
                  rules={[]}
                >
                  <Input.Password
                    placeholder="Leave blank to keep existing"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      AWS Secret Access Key
                      <Tooltip title="Optional. Required if AWS Access Key ID is provided.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "aws_secret_access_key"]}
                  rules={[]}
                >
                  <Input.Password
                    placeholder="Leave blank to keep existing"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      AWS Session Token
                      <Tooltip title="Optional. Only needed for temporary STS credentials.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "aws_session_token"]}
                >
                  <Input.Password
                    placeholder="Leave blank to keep existing"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      AWS Role ARN
                      <Tooltip title="Optional. IAM role ARN to assume via STS before signing. If set, LiteLLM calls sts:AssumeRole to get temporary credentials.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "aws_role_name"]}
                >
                  <Input
                    placeholder="Leave blank to keep existing"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      AWS Session Name
                      <Tooltip title="Optional. Session name for the AssumeRole call — appears in CloudTrail logs. Auto-generated if omitted.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "aws_session_name"]}
                >
                  <Input
                    placeholder="Leave blank to keep existing"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
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
                  server_name: currentServerName ?? mcpServer.server_name,
                  url: currentUrl ?? mcpServer.url,
                  spec_path: currentSpecPath ?? mcpServer.spec_path,
                  transport: transportType ?? mcpServer.transport,
                  auth_type: currentAuthType ?? mcpServer.auth_type,
                  mcp_info: mcpServer.mcp_info,
                  oauth_flow_type: (currentTokenUrl ?? mcpServer.token_url) ? OAUTH_FLOW.M2M : OAUTH_FLOW.INTERACTIVE,
                  static_headers: currentStaticHeaders ?? mcpServer.static_headers,
                  credentials: currentCredentials,
                  authorization_url: currentAuthorizationUrl ?? mcpServer.authorization_url,
                  token_url: currentTokenUrl ?? mcpServer.token_url,
                  registration_url: currentRegistrationUrl ?? mcpServer.registration_url,
                }}
                allowedTools={allowedTools}
                existingAllowedTools={mcpServer.allowed_tools || null}
                onAllowedToolsChange={setAllowedTools}
                toolNameToDisplayName={toolNameToDisplayName}
                toolNameToDescription={toolNameToDescription}
                onToolNameToDisplayNameChange={setToolNameToDisplayName}
                onToolNameToDescriptionChange={setToolNameToDescription}
                externalTools={tools}
                externalIsLoading={isLoadingTools}
                externalError={toolsError}
                externalCanFetch={!!mcpServer.server_id}
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
