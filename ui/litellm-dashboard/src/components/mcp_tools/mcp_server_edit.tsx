import React, { useState, useEffect } from "react";
import { Form, Select, Button as AntdButton, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, TextInput, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { AUTH_TYPE, MCPServer, MCPServerCostInfo } from "./types";
import { updateMCPServer, testMCPToolsListRequest } from "../networking";
import MCPServerCostConfig from "./mcp_server_cost_config";
import MCPPermissionManagement from "./MCPPermissionManagement";
import MCPToolConfiguration from "./mcp_tool_configuration";
import { validateMCPServerUrl, validateMCPServerName } from "./utils";
import NotificationsManager from "../molecules/notifications_manager";

interface MCPServerEditProps {
  mcpServer: MCPServer;
  accessToken: string | null;
  onCancel: () => void;
  onSuccess: (server: MCPServer) => void;
  availableAccessGroups: string[];
}

const AUTH_TYPES_REQUIRING_AUTH_VALUE = [AUTH_TYPE.API_KEY, AUTH_TYPE.BEARER_TOKEN, AUTH_TYPE.BASIC];

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
  const authType = Form.useWatch("auth_type", form) as string | undefined;
  const shouldShowAuthValueField = authType ? AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(authType) : false;

  const initialStaticHeaders = React.useMemo(() => {
    if (!mcpServer.static_headers) {
      return [];
    }
    return Object.entries(mcpServer.static_headers).map(([header, value]) => ({
      header,
      value: value != null ? String(value) : "",
    }));
  }, [mcpServer.static_headers]);

  const initialValues = React.useMemo(
    () => ({
      ...mcpServer,
      static_headers: initialStaticHeaders,
    }),
    [mcpServer, initialStaticHeaders],
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
  }, [mcpServer, accessToken]);

  const fetchTools = async () => {
    if (!accessToken || !mcpServer.url) {
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
      };

      const toolsResponse = await testMCPToolsListRequest(accessToken, mcpServerConfig);

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

  const handleSave = async (values: Record<string, any>) => {
    if (!accessToken) return;
    try {
      // Ensure access groups is always a string array
      const { static_headers: staticHeadersList, credentials: credentialValues, ...restValues } = values;

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

      // Prepare the payload with cost configuration and permission fields
      const payload: Record<string, any> = {
        ...restValues,
        server_id: mcpServer.server_id,
        mcp_info: {
          server_name: restValues.server_name || restValues.url,
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
      };

      const includeCredentials = restValues.auth_type && AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(restValues.auth_type);

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
              <TextInput />
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
              <TextInput onChange={() => setAliasManuallyEdited(true)} />
            </Form.Item>
            <Form.Item label="Description" name="description">
              <TextInput />
            </Form.Item>
            <Form.Item
              label="MCP Server URL"
              name="url"
              rules={[
                { required: true, message: "Please enter a server URL" },
                { validator: (_, value) => validateMCPServerUrl(value) },
              ]}
            >
              <TextInput />
            </Form.Item>
            <Form.Item label="Transport Type" name="transport" rules={[{ required: true }]}>
              <Select>
                <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                <Select.Option value="http">HTTP</Select.Option>
              </Select>
            </Form.Item>
            <Form.Item label="Authentication" name="auth_type" rules={[{ required: true }]}>
              <Select>
                <Select.Option value="none">None</Select.Option>
                <Select.Option value="api_key">API Key</Select.Option>
                <Select.Option value="bearer_token">Bearer Token</Select.Option>
                <Select.Option value="basic">Basic Auth</Select.Option>
              </Select>
            </Form.Item>

            {shouldShowAuthValueField && (
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
                <TextInput
                  type="password"
                  placeholder="Enter token or secret (leave blank to keep existing)"
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>
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
                formValues={{
                  server_id: mcpServer.server_id,
                  server_name: mcpServer.server_name,
                  url: mcpServer.url,
                  transport: mcpServer.transport,
                  auth_type: mcpServer.auth_type,
                  mcp_info: mcpServer.mcp_info,
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
