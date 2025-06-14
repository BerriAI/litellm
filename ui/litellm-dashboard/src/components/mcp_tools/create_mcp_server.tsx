import React, { useState } from "react";
import {
  Modal,
  Tooltip,
  Form,
  Select,
  message,
  Button as AntdButton,
} from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, TextInput } from "@tremor/react";
import { createMCPServer } from "../networking";
import { MCPServer } from "./types";
import { isAdminRole } from "@/utils/roles";


const asset_logos_folder = '../ui/assets/logos/';
export const mcpLogoImg = `${asset_logos_folder}mcp_logo.png`;

interface CreateMCPServerProps {
  userRole: string;
  accessToken: string | null;
  onCreateSuccess: (newMcpServer: MCPServer) => void;
}

const CreateMCPServer: React.FC<CreateMCPServerProps> = ({
  userRole,
  accessToken,
  onCreateSuccess,
}) => {
  const [form] = Form.useForm();
  const [isLoading, setIsLoading] = useState(false);

  const handleCreate = async (formValues: Record<string, any>) => {
    setIsLoading(true);
    try {
      console.log(`formValues: ${JSON.stringify(formValues)}`);

      if (accessToken != null) {
        const response: MCPServer = await createMCPServer(
          accessToken,
          formValues
        );

        message.success("MCP Server created successfully");
        form.resetFields();
        setModalVisible(false);
        onCreateSuccess(response);
      }
    } catch (error) {
      message.error("Error creating MCP Server: " + error, 20);
    } finally {
      setIsLoading(false);
    }
  };

  // state
  const [isModalVisible, setModalVisible] = useState(false);

  const handleCancel = () => {
    form.resetFields();
    setModalVisible(false);
  };

  // rendering
  if (!isAdminRole(userRole)) {
    return null;
  }

  return (
    <div>
      <Button 
        className="mx-auto mb-4" 
        onClick={() => setModalVisible(true)}
      >
        + Add New MCP Server
      </Button>

      <Modal
        title={
          <div className="flex items-center space-x-3 pb-4 border-b border-gray-100">
              <img 
                src={mcpLogoImg}
                alt="MCP Logo" 
                className="w-8 h-8 object-contain"
                style={{ 
                  height: '20px', 
                  width: '20px', 
                  marginRight: '8px',
                  objectFit: 'contain'
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
          body: { padding: '24px' },
          header: { padding: '24px 24px 0 24px', border: 'none' },
        }}
      >
        <div className="mt-6">
          <Form
            form={form}
            onFinish={handleCreate}
            layout="vertical"
            className="space-y-6"
          >
            <div className="grid grid-cols-1 gap-6">
            <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    MCP Server Name
                    <Tooltip title="Best practice: Use a descriptive name that indicates the server's purpose (e.g., 'GitHub Integration', 'Email Service')">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="alias"
                rules={[
                  { required: false, message: "Please enter a server name" },
                ]}
              >
                <TextInput 
                  placeholder="e.g., GitHub MCP, Zapier MCP, etc." 
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>

              <Form.Item
                label={<span className="text-sm font-medium text-gray-700">Description</span>}
                name="description"
                rules={[
                  {
                    required: false,
                    message: "Please enter a server description",
                  },
                ]}
              >
                <TextInput 
                  placeholder="Brief description of what this server does" 
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700">
                    MCP Server URL <span className="text-red-500">*</span>
                  </span>
                }
                name="url"
                rules={[
                  { required: true, message: "Please enter a server URL" },
                  { type: 'url', message: "Please enter a valid URL" }
                ]}
              >
                <TextInput 
                  placeholder="https://your-mcp-server.com" 
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>

              <div className="grid grid-cols-2 gap-4">
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Transport Type <span className="text-red-500">*</span>
                    </span>
                  }
                  name="transport"
                  rules={[{ required: true, message: "Please select a transport type" }]}
                >
                  <Select 
                    placeholder="Select transport"
                    className="rounded-lg"
                    size="large"
                  >
                    <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                    <Select.Option value="http">HTTP</Select.Option>
                  </Select>
                </Form.Item>

                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Authentication <span className="text-red-500">*</span>
                    </span>
                  }
                  name="auth_type"
                  rules={[{ required: true, message: "Please select an auth type" }]}
                >
                  <Select 
                    placeholder="Select auth type"
                    className="rounded-lg"
                    size="large"
                  >
                    <Select.Option value="none">None</Select.Option>
                    <Select.Option value="api_key">API Key</Select.Option>
                    <Select.Option value="bearer_token">Bearer Token</Select.Option>
                    <Select.Option value="basic">Basic Auth</Select.Option>
                  </Select>
                </Form.Item>
              </div>

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    MCP Version <span className="text-red-500 ml-1">*</span>
                    <Tooltip title="Select the MCP specification version your server supports">
                      <InfoCircleOutlined className="ml-2 text-gray-400 hover:text-gray-600" />
                    </Tooltip>
                  </span>
                }
                name="spec_version"
                rules={[
                  { required: true, message: "Please select a spec version" },
                ]}
              >
                <Select 
                  placeholder="Select MCP version"
                  className="rounded-lg"
                  size="large"
                >
                  <Select.Option value="2025-03-26">2025-03-26 (Latest)</Select.Option>
                  <Select.Option value="2024-11-05">2024-11-05</Select.Option>
                </Select>
              </Form.Item>
            </div>

            <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
              <Button 
                variant="secondary"
                onClick={handleCancel}
              >
                Cancel
              </Button>
              <Button 
                variant="primary"
                loading={isLoading}
              >
                {isLoading ? 'Creating...' : 'Add MCP Server'}
              </Button>
            </div>
          </Form>
        </div>
      </Modal>
    </div>
  );
};

export default CreateMCPServer; 