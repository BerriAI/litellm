import React from "react";
import { Form, Select, Button as AntdButton, message } from "antd";
import { Button, TextInput } from "@tremor/react";
import { MCPServer } from "./types";
import { updateMCPServer } from "../networking";

interface MCPServerEditProps {
  mcpServer: MCPServer;
  accessToken: string | null;
  onCancel: () => void;
  onSuccess: (server: MCPServer) => void;
}

const MCPServerEdit: React.FC<MCPServerEditProps> = ({ mcpServer, accessToken, onCancel, onSuccess }) => {
  const [form] = Form.useForm();

  const handleSave = async (values: Record<string, any>) => {
    if (!accessToken) return;
    try {
      const updated = await updateMCPServer(accessToken, { ...values, server_id: mcpServer.server_id });
      message.success("MCP Server updated successfully");
      onSuccess(updated);
    } catch (error: any) {
      message.error("Failed to update MCP Server" + (error?.message ? `: ${error.message}` : ""));
    }
  };

  return (
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
  );
};

export default MCPServerEdit;
