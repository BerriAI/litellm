import React, { useState, useEffect } from "react";
import { Form, Select, Button as AntdButton, message, Input, Space, Tooltip } from "antd";
import { Button, TextInput, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { MCPServer, MCPServerCostInfo } from "./types";
import { updateMCPServer, testMCPToolsListRequest } from "../networking";
import MCPServerCostConfig from "./mcp_server_cost_config";
import { MinusCircleOutlined, PlusOutlined, InfoCircleOutlined } from "@ant-design/icons";

interface MCPServerEditProps {
  mcpServer: MCPServer;
  accessToken: string | null;
  onCancel: () => void;
  onSuccess: (server: MCPServer) => void;
  availableAccessGroups: string[];
}

const MCPServerEdit: React.FC<MCPServerEditProps> = ({ mcpServer, accessToken, onCancel, onSuccess, availableAccessGroups }) => {
  const [form] = Form.useForm();
  const [costConfig, setCostConfig] = useState<MCPServerCostInfo>({});
  const [tools, setTools] = useState<any[]>([]);
  const [isLoadingTools, setIsLoadingTools] = useState(false);
  const [searchValue, setSearchValue] = useState<string>("");

  // Initialize cost config from existing server data
  useEffect(() => {
    if (mcpServer.mcp_info?.mcp_server_cost_info) {
      setCostConfig(mcpServer.mcp_info.mcp_server_cost_info);
    }
  }, [mcpServer]);

  // Transform string array to object array for initial form values
  useEffect(() => {
    if (mcpServer.mcp_access_groups) {
      // If access groups are objects, extract the name property; if strings, use as is
      const groupNames = mcpServer.mcp_access_groups.map((g: any) => typeof g === 'string' ? g : g.name || String(g));
      form.setFieldValue('mcp_access_groups', groupNames);
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
        alias: mcpServer.alias,
        url: mcpServer.url,
        transport: mcpServer.transport,
        spec_version: mcpServer.spec_version,
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
    }))

    // If search value doesn't match any existing group and is not empty, add "create new group" option
    if (searchValue && !availableAccessGroups.some(group => group.toLowerCase().includes(searchValue.toLowerCase()))) {
      existingOptions.push({
        value: searchValue,
        label: (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span className="font-medium">{searchValue}</span>
            <span className="text-gray-400 text-xs ml-1">create new group</span>
          </div>
        ),
      })
    }

    return existingOptions
  }

  const handleSave = async (values: Record<string, any>) => {
    if (!accessToken) return;
    try {
      // Ensure access groups is always a string array
      const accessGroups = (values.mcp_access_groups || []).map((g: any) => typeof g === 'string' ? g : g.name || String(g));

      // Prepare the payload with cost configuration
      const payload = {
        ...values,
        server_id: mcpServer.server_id,
        mcp_info: {
          server_name: values.alias || values.url,
          description: values.description,
          mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null
        },
        mcp_access_groups: accessGroups
      };

      const updated = await updateMCPServer(accessToken, payload);
      message.success("MCP Server updated successfully");
      onSuccess(updated);
    } catch (error: any) {
      message.error("Failed to update MCP Server" + (error?.message ? `: ${error.message}` : ""));
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
          <Form form={form} onFinish={handleSave} initialValues={mcpServer} layout="vertical">
            <Form.Item label="MCP Server Name" name="alias" rules={[{
              validator: (_, value) =>
                value && value.includes('-')
                  ? Promise.reject("Server name cannot contain '-' (hyphen). Please use '_' (underscore) instead.")
                  : Promise.resolve(),
            }]}>
              <TextInput />
            </Form.Item>
            <Form.Item label="Description" name="description">
              <TextInput />
            </Form.Item>
            <Form.Item label="MCP Server URL" name="url" rules={[{ required: true, message: "Please enter a server URL" }]}> 
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
            <Form.Item label="MCP Version" name="spec_version" rules={[{ required: true }]}> 
              <Select>
                <Select.Option value="2025-03-26">2025-03-26 (Latest)</Select.Option>
                <Select.Option value="2024-11-05">2024-11-05</Select.Option>
              </Select>
            </Form.Item>

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  MCP Access Groups
                  <Tooltip title="Define access groups for this MCP server. Each group represents a set of permissions.">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="mcp_access_groups"
              getValueFromEvent={value => value}
            >
              <Select
                mode="tags"
                style={{ width: '100%' }}
                showSearch
                placeholder="Add or select access groups"
                tokenSeparators={[',']}
                optionFilterProp="value"
                filterOption={(input, option) =>
                  (option?.value ?? '').toLowerCase().includes(input.toLowerCase())
                }
                onSearch={(value) => setSearchValue(value)}
                options={getAccessGroupOptions()}
                // Ensure value is always an array of strings
                getPopupContainer={trigger => trigger.parentNode}
              />
            </Form.Item>

            <div className="flex justify-end gap-2">
              <AntdButton onClick={onCancel}>Cancel</AntdButton>
              <Button type="submit">Save Changes</Button>
            </div>
          </Form>
        </TabPanel>
        
        <TabPanel>
          <div className="space-y-6">
              <MCPServerCostConfig
                value={costConfig}
                onChange={setCostConfig}
                tools={tools}
                disabled={isLoadingTools}
              />
            
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
