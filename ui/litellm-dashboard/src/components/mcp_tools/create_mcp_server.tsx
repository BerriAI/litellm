import React, { useState } from "react";
import { Modal, Tooltip, Form, Select } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, TextInput } from "@tremor/react";
import { createMCPServer } from "../networking";
import { MCPServer, MCPServerCostInfo } from "./types";
import MCPServerCostConfig from "./mcp_server_cost_config";
import MCPConnectionStatus from "./mcp_connection_status";
import MCPToolConfiguration from "./mcp_tool_configuration";
import StdioConfiguration from "./StdioConfiguration";
import MCPPermissionManagement from "./MCPPermissionManagement";
import { isAdminRole } from "@/utils/roles";
import { validateMCPServerUrl, validateMCPServerName } from "./utils";
import NotificationsManager from "../molecules/notifications_manager";

const asset_logos_folder = "../ui/assets/logos/";
export const mcpLogoImg = `${asset_logos_folder}mcp_logo.png`;

interface CreateMCPServerProps {
  userRole: string;
  accessToken: string | null;
  onCreateSuccess: (newMcpServer: MCPServer) => void;
  isModalVisible: boolean;
  setModalVisible: (visible: boolean) => void;
  availableAccessGroups: string[];
}

const CreateMCPServer: React.FC<CreateMCPServerProps> = ({
  userRole,
  accessToken,
  onCreateSuccess,
  isModalVisible,
  setModalVisible,
  availableAccessGroups,
}) => {
  const [form] = Form.useForm();
  const [isLoading, setIsLoading] = useState(false);
  const [costConfig, setCostConfig] = useState<MCPServerCostInfo>({});
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [aliasManuallyEdited, setAliasManuallyEdited] = useState(false);
  const [tools, setTools] = useState<any[]>([]);
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [transportType, setTransportType] = useState<string>("");
  const [searchValue, setSearchValue] = useState<string>("");
  const [urlWarning, setUrlWarning] = useState<string>("");

  // Function to check URL format based on transport type
  const checkUrlFormat = (url: string, transport: string) => {
    if (!url) {
      setUrlWarning("");
      return;
    }

    if (transport === "sse" && !url.endsWith("/sse")) {
      setUrlWarning("Typically MCP SSE URLs end with /sse. You can add this url but this is a warning.");
    } else if (transport === "http" && !url.endsWith("/mcp")) {
      setUrlWarning("Typically MCP HTTP URLs end with /mcp. You can add this url but this is a warning.");
    } else {
      setUrlWarning("");
    }
  };

  const handleCreate = async (formValues: Record<string, any>) => {
    setIsLoading(true);
    try {
      // Transform access groups into objects with name property

      const accessGroups = formValues.mcp_access_groups;

      // Process stdio configuration if present
      let stdioFields = {};
      if (formValues.stdio_config && transportType === "stdio") {
        try {
          const stdioConfig = JSON.parse(formValues.stdio_config);

          // Handle both formats:
          // 1. Full mcpServers structure: {"mcpServers": {"server-name": {...}}}
          // 2. Direct config: {"command": "...", "args": [...], "env": {...}}

          let actualConfig = stdioConfig;

          // If it's the full mcpServers structure, extract the first server config
          if (stdioConfig.mcpServers && typeof stdioConfig.mcpServers === "object") {
            const serverNames = Object.keys(stdioConfig.mcpServers);
            if (serverNames.length > 0) {
              const firstServerName = serverNames[0];
              actualConfig = stdioConfig.mcpServers[firstServerName];

              // If no alias is provided, use the server name from the JSON
              if (!formValues.server_name) {
                formValues.server_name = firstServerName.replace(/-/g, "_"); // Replace hyphens with underscores
              }
            }
          }

          stdioFields = {
            command: actualConfig.command,
            args: actualConfig.args,
            env: actualConfig.env,
          };

          console.log("Parsed stdio config:", stdioFields);
        } catch (error) {
          NotificationsManager.fromBackend("Invalid JSON in stdio configuration");
          return;
        }
      }

      // Prepare the payload with cost configuration and allowed tools
      const payload = {
        ...formValues,
        ...stdioFields,
        // Remove the raw stdio_config field as we've extracted its components
        stdio_config: undefined,
        mcp_info: {
          server_name: formValues.server_name || formValues.url,
          description: formValues.description,
          mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null,
        },
        mcp_access_groups: accessGroups,
        alias: formValues.alias,
        allowed_tools: allowedTools.length > 0 ? allowedTools : null,
      };

      console.log(`Payload: ${JSON.stringify(payload)}`);

      if (accessToken != null) {
        const response = await createMCPServer(accessToken, payload);

        NotificationsManager.success("MCP Server created successfully");
        form.resetFields();
        setCostConfig({});
        setTools([]);
        setAllowedTools([]);
        setUrlWarning("");
        setAliasManuallyEdited(false);
        setModalVisible(false);
        onCreateSuccess(response);
      }
    } catch (error) {
      NotificationsManager.fromBackend("Error creating MCP Server: " + error);
    } finally {
      setIsLoading(false);
    }
  };

  // state
  const handleCancel = () => {
    form.resetFields();
    setCostConfig({});
    setTools([]);
    setAllowedTools([]);
    setUrlWarning("");
    setAliasManuallyEdited(false);
    setModalVisible(false);
  };

  const handleTransportChange = (value: string) => {
    setTransportType(value);
    // Clear fields that are not relevant for the selected transport
    if (value === "stdio") {
      form.setFieldsValue({ url: undefined, auth_type: undefined });
      setUrlWarning("");
    } else {
      form.setFieldsValue({ command: undefined, args: undefined, env: undefined });
      // Check URL format for the new transport type
      const currentUrl = form.getFieldValue("url");
      if (currentUrl) {
        checkUrlFormat(currentUrl, value);
      }
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

  // Auto-populate alias from server_name unless manually edited
  React.useEffect(() => {
    if (!aliasManuallyEdited && formValues.server_name) {
      const normalized = formValues.server_name.replace(/\s+/g, "_");
      form.setFieldsValue({ alias: normalized });
      setFormValues((prev) => ({ ...prev, alias: normalized }));
    }
  }, [formValues.server_name]);

  // Clear formValues when modal closes to reset child components
  React.useEffect(() => {
    if (!isModalVisible) {
      setFormValues({});
    }
  }, [isModalVisible]);

  // rendering
  if (!isAdminRole(userRole)) {
    return null;
  }

  return (
    <Modal
      title={
        <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
          <img
            src={mcpLogoImg}
            alt="MCP Logo"
            className="w-8 h-8 object-contain"
            style={{
              height: "20px",
              width: "20px",
              marginRight: "8px",
              objectFit: "contain",
            }}
          />
          <h2 className="text-xl font-semibold text-gray-900">Add New MCP Server</h2>
        </div>
      }
      open={isModalVisible}
      width={1000}
      onCancel={handleCancel}
      footer={null}
      className="top-8"
      styles={{
        body: { padding: "24px" },
        header: { padding: "24px 24px 0 24px", border: "none" },
      }}
    >
      <div className="mt-6">
        <Form
          form={form}
          onFinish={handleCreate}
          onValuesChange={(_, allValues) => setFormValues(allValues)}
          layout="vertical"
          className="space-y-6"
        >
          <div className="grid grid-cols-1 gap-6">
            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  MCP Server Name
                  <Tooltip title="Best practice: Use a descriptive name that indicates the server's purpose (e.g., 'GitHub_MCP', 'Email_Service'). Hyphens '-' are not allowed; use underscores '_' instead.">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="server_name"
              rules={[
                { required: false, message: "Please enter a server name" },
                { validator: (_, value) => validateMCPServerName(value) },
              ]}
            >
              <TextInput
                placeholder="e.g., GitHub_MCP, Zapier_MCP, etc."
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  Alias
                  <Tooltip title="A short, unique identifier for this server. Defaults to the server name with spaces replaced by underscores.">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="alias"
              rules={[
                { required: false },
                {
                  validator: (_, value) =>
                    value && value.includes("-")
                      ? Promise.reject("Alias cannot contain '-' (hyphen). Please use '_' (underscore) instead.")
                      : Promise.resolve(),
                },
              ]}
            >
              <TextInput
                placeholder="e.g., GitHub_MCP, Zapier_MCP, etc."
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                onChange={() => setAliasManuallyEdited(true)}
              />
            </Form.Item>

            <Form.Item
              label={<span className="text-sm font-medium text-gray-700">Description</span>}
              name="description"
              rules={[
                {
                  required: false,
                  message: "Please enter a server description!!!!!!!!!",
                },
              ]}
            >
              <TextInput
                placeholder="Brief description of what this server does"
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>

            <Form.Item
              label={<span className="text-sm font-medium text-gray-700">Transport Type</span>}
              name="transport"
              rules={[{ required: true, message: "Please select a transport type" }]}
            >
              <Select
                placeholder="Select transport"
                className="rounded-lg"
                size="large"
                onChange={handleTransportChange}
                value={transportType}
              >
                <Select.Option value="http">HTTP</Select.Option>
                <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                <Select.Option value="stdio">Standard Input/Output (stdio)</Select.Option>
              </Select>
            </Form.Item>

            {/* URL field - only show for HTTP and SSE */}
            {transportType !== "stdio" && (
              <Form.Item
                label={<span className="text-sm font-medium text-gray-700">MCP Server URL</span>}
                name="url"
                rules={[
                  { required: true, message: "Please enter a server URL" },
                  { validator: (_, value) => validateMCPServerUrl(value) },
                ]}
              >
                <div>
                  <TextInput
                    placeholder="https://your-mcp-server.com"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                    onChange={(e) => checkUrlFormat(e.target.value, transportType)}
                  />
                  {urlWarning && <div className="mt-1 text-red-500 text-sm font-medium">{urlWarning}</div>}
                </div>
              </Form.Item>
            )}

            {/* Authentication - only show for HTTP and SSE */}
            {transportType !== "stdio" && (
              <Form.Item
                label={<span className="text-sm font-medium text-gray-700">Authentication</span>}
                name="auth_type"
                rules={[{ required: true, message: "Please select an auth type" }]}
              >
                <Select placeholder="Select auth type" className="rounded-lg" size="large">
                  <Select.Option value="none">None</Select.Option>
                  <Select.Option value="api_key">API Key</Select.Option>
                  <Select.Option value="bearer_token">Bearer Token</Select.Option>
                  <Select.Option value="basic">Basic Auth</Select.Option>
                </Select>
              </Form.Item>
            )}

            {/* Stdio Configuration - only show for stdio transport */}
            <StdioConfiguration isVisible={transportType === "stdio"} />
          </div>

          {/* Permission Management / Access Control Section */}
          <div className="mt-8">
            <MCPPermissionManagement
              availableAccessGroups={availableAccessGroups}
              mcpServer={null}
              searchValue={searchValue}
              setSearchValue={setSearchValue}
              getAccessGroupOptions={getAccessGroupOptions}
            />
          </div>

          {/* Connection Status Section */}
          <div className="mt-8 pt-6 border-t border-gray-200">
            <MCPConnectionStatus accessToken={accessToken} formValues={formValues} onToolsLoaded={setTools} />
          </div>

          {/* Tool Configuration Section */}
          <div className="mt-6">
            <MCPToolConfiguration
              accessToken={accessToken}
              formValues={formValues}
              allowedTools={allowedTools}
              existingAllowedTools={null}
              onAllowedToolsChange={setAllowedTools}
            />
          </div>

          {/* Cost Configuration Section */}
          <div className="mt-6">
            <MCPServerCostConfig
              value={costConfig}
              onChange={setCostConfig}
              tools={tools.filter((tool) => allowedTools.includes(tool.name))}
              disabled={false}
            />
          </div>

          <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
            <Button variant="secondary" onClick={handleCancel}>
              Cancel
            </Button>
            <Button variant="primary" loading={isLoading}>
              {isLoading ? "Creating..." : "Add MCP Server"}
            </Button>
          </div>
        </Form>
      </div>
    </Modal>
  );
};

export default CreateMCPServer;
