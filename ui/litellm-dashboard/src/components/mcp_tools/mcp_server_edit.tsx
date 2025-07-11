import React, { useState, useEffect } from "react";
import { Form, Select, Button as AntdButton, message } from "antd";
import { Button, TextInput, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { MCPServer, MCPServerCostInfo } from "./types";
import { updateMCPServer, testMCPToolsListRequest } from "../networking";
import MCPServerCostConfig from "./mcp_server_cost_config";

interface MCPServerEditProps {
  mcpServer: MCPServer;
  accessToken: string | null;
  onCancel: () => void;
  onSuccess: (server: MCPServer) => void;
}

const MCPServerEdit: React.FC<MCPServerEditProps> = ({ mcpServer, accessToken, onCancel, onSuccess }) => {
  const [form] = Form.useForm();
  const [costConfig, setCostConfig] = useState<MCPServerCostInfo>({});
  const [tools, setTools] = useState<any[]>([]);
  const [isLoadingTools, setIsLoadingTools] = useState(false);

  // Initialize cost config from existing server data
  useEffect(() => {
    if (mcpServer.mcp_info?.mcp_server_cost_info) {
      setCostConfig(mcpServer.mcp_info.mcp_server_cost_info);
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

  const handleSave = async (values: Record<string, any>) => {
    if (!accessToken) return;
    try {
      // Prepare the payload with cost configuration
      const payload = {
        ...values,
        server_id: mcpServer.server_id,
        mcp_info: {
          server_name: values.alias || values.url,
          description: values.description,
          mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null
        }
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
            <Form.Item label="MCP Server Name" name="alias">
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
