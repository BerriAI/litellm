import React, { useState } from "react";
import {
  Modal,
  Tooltip,
  Form,
  Select,
  message,
  Button as AntdButton,
  Space,
  Input,
} from "antd";
import { InfoCircleOutlined, MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { Button, TextInput } from "@tremor/react";
import { createMCPServer } from "../networking";
import { MCPServer, MCPServerCostInfo } from "./types";
import MCPServerCostConfig from "./mcp_server_cost_config";
import MCPConnectionStatus from "./mcp_connection_status";
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
  const [costConfig, setCostConfig] = useState<MCPServerCostInfo>({});
  const [mcpAccessGroups, setMcpAccessGroups] = useState<string[]>([]);
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [tools, setTools] = useState<any[]>([]);

  const handleCreate = async (formValues: Record<string, any>) => {
    setIsLoading(true);
    try {
      // Transform access groups into objects with name property
      
      const accessGroups = formValues.mcp_access_groups

      // Prepare the payload with cost configuration
      let payload: Record<string, any> = {
        ...formValues,
        mcp_info: {
          server_name: formValues.alias || formValues.url || formValues.command,
          description: formValues.description,
          mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null,
        },
        mcp_access_groups: accessGroups
      };

      // Handle stdio-specific fields
      if (formValues.transport === "stdio") {
        // Transform environment variables from form format to object format
        if (formValues.env) {
          const envObj: Record<string, string> = {};
          formValues.env.forEach((envVar: any) => {
            if (envVar.key && envVar.value) {
              envObj[envVar.key] = envVar.value;
            }
          });
          payload.env = envObj;
        }
        // Remove URL for stdio transport
        delete payload.url;
      } else {
        // Remove stdio fields for non-stdio transport
        delete payload.command;
        delete payload.args;
        delete payload.env;
      }

      console.log(`Payload: ${JSON.stringify(payload)}`);

      if (accessToken != null) {
        const response = await createMCPServer(
          accessToken,
          payload
        );

        message.success("MCP Server created successfully");
        form.resetFields();
        setCostConfig({});
        setTools([]);
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
    setCostConfig({});
    setTools([]);
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
                name="alias"
                rules={[
                  { required: false, message: "Please enter a server name" },
                  {
                    validator: (_, value) =>
                      value && value.includes('-')
                        ? Promise.reject("Server name cannot contain '-' (hyphen). Please use '_' (underscore) instead.")
                        : Promise.resolve(),
                  },
                ]}
              >
                <TextInput 
                  placeholder="e.g., GitHub_MCP, Zapier_MCP, etc." 
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

              {formValues.transport !== "stdio" && (
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      MCP Server URL
                    </span>
                  }
                  name="url"
                  rules={[
                    { required: formValues.transport !== "stdio", message: "Please enter a server URL" },
                    { type: 'url', message: "Please enter a valid URL" }
                  ]}
                >
                  <TextInput 
                    placeholder="https://your-mcp-server.com" 
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
              )}

              <div className="grid grid-cols-2 gap-4">
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Transport Type
                    </span>
                  }
                  name="transport"
                  rules={[{ required: true, message: "Please select a transport type" }]}
                >
                  <Select 
                    placeholder="Select transport"
                    className="rounded-lg"
                    size="large"
                    onChange={(value) => setFormValues({...formValues, transport: value})}
                  >
                    <Select.Option value="http">HTTP</Select.Option>
                    <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                    <Select.Option value="stdio">Standard I/O (stdio)</Select.Option>
                  </Select>
                </Form.Item>

                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700">
                      Authentication
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

              {/* Stdio-specific fields */}
              {formValues.transport === "stdio" && (
                <>
                  <Form.Item
                    label={
                      <span className="text-sm font-medium text-gray-700 flex items-center">
                        Command
                        <Tooltip title="The command to run the MCP server (e.g., 'npx', 'python', 'node')">
                          <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                        </Tooltip>
                      </span>
                    }
                    name="command"
                    rules={[
                      { required: formValues.transport === "stdio", message: "Please enter a command" },
                    ]}
                  >
                    <TextInput 
                      placeholder="e.g., npx, python, node" 
                      className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                    />
                  </Form.Item>

                  <Form.Item
                    label={
                      <span className="text-sm font-medium text-gray-700 flex items-center">
                        Arguments
                        <Tooltip title="Arguments to pass to the command. Each argument should be on a separate line.">
                          <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                        </Tooltip>
                      </span>
                    }
                    name="args"
                    rules={[
                      { required: formValues.transport === "stdio", message: "Please enter arguments" },
                    ]}
                  >
                    <Form.List name="args">
                      {(fields, { add, remove }) => (
                        <>
                          {fields.map(({ key, name, ...restField }) => (
                            <Space key={key} style={{ display: 'flex', marginBottom: 8 }} align="baseline">
                              <Form.Item
                                {...restField}
                                name={[name]}
                                rules={[{ required: true, message: 'Missing argument' }]}
                              >
                                <Input placeholder="e.g., -y, @circleci/mcp-server-circleci" />
                              </Form.Item>
                              <MinusCircleOutlined onClick={() => remove(name)} />
                            </Space>
                          ))}
                          <Form.Item>
                            <AntdButton type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                              Add argument
                            </AntdButton>
                          </Form.Item>
                        </>
                      )}
                    </Form.List>
                  </Form.Item>

                  <Form.Item
                    label={
                      <span className="text-sm font-medium text-gray-700 flex items-center">
                        Environment Variables
                        <Tooltip title="Environment variables to set when running the command. Format: KEY=VALUE">
                          <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                        </Tooltip>
                      </span>
                    }
                    name="env"
                  >
                    <Form.List name="env">
                      {(fields, { add, remove }) => (
                        <>
                          {fields.map(({ key, name, ...restField }) => (
                            <Space key={key} style={{ display: 'flex', marginBottom: 8 }} align="baseline">
                              <Form.Item
                                {...restField}
                                name={[name, 'key']}
                                rules={[{ required: true, message: 'Missing environment variable key' }]}
                              >
                                <Input placeholder="e.g., CIRCLECI_TOKEN" />
                              </Form.Item>
                              <Form.Item
                                {...restField}
                                name={[name, 'value']}
                                rules={[{ required: true, message: 'Missing environment variable value' }]}
                              >
                                <Input placeholder="e.g., your-token-value" />
                              </Form.Item>
                              <MinusCircleOutlined onClick={() => remove(name)} />
                            </Space>
                          ))}
                          <Form.Item>
                            <AntdButton type="dashed" onClick={() => add()} block icon={<PlusOutlined />}>
                              Add environment variable
                            </AntdButton>
                          </Form.Item>
                        </>
                      )}
                    </Form.List>
                  </Form.Item>
                </>
              )}

              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    MCP Version
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

              <Form.Item
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
                <Select
                  mode="tags"
                  showSearch
                  placeholder="Select existing groups or type to create new ones"
                  optionFilterProp="children"
                  tokenSeparators={[',']}
                  options={mcpAccessGroups.map((group) => ({
                    value: group,
                    label: group
                  }))}
                  maxTagCount="responsive"
                  allowClear
                />
              </Form.Item>
            </div>

            {/* Cost Configuration Section */}
            <div className="mt-8 pt-6 border-t border-gray-200">
              <MCPServerCostConfig
                value={costConfig}
                onChange={setCostConfig}
                tools={tools}
                disabled={false}
              />
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