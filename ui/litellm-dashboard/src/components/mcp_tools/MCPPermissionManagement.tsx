import React, { useEffect } from "react";
import { Form, Select, Tooltip, Collapse, Input, Space, Button } from "antd";
import { InfoCircleOutlined, MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { MCPServer } from "./types";
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
  const form = Form.useFormInstance();

  // Set initial values when mcpServer changes
  useEffect(() => {
    if (mcpServer) {
      // Set extra_headers if they exist
      if (mcpServer.extra_headers) {
        form.setFieldValue("extra_headers", mcpServer.extra_headers);
      }
      if (mcpServer.static_headers) {
        const staticHeaders = Object.entries(mcpServer.static_headers).map(([header, value]) => ({
          header,
          value: value != null ? String(value) : "",
        }));
        form.setFieldValue("static_headers", staticHeaders);
      }
    }
  }, [mcpServer, form]);

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
      >
        <div className="space-y-6 pt-4">
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
              optionFilterProp="value"
              filterOption={(input, option) => (option?.value ?? "").toLowerCase().includes(input.toLowerCase())}
              onSearch={(value) => setSearchValue(value)}
              tokenSeparators={[","]}
              options={getAccessGroupOptions()}
              maxTagCount="responsive"
              allowClear
            />
          </Form.Item>

          <Form.Item
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
            <Select
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
          </Form.Item>

          <Form.Item
            label={
              <span className="text-sm font-medium text-gray-700 flex items-center">
                Static Headers
                <Tooltip title="Send these key-value headers with every request to this MCP server.">
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
              </span>
            }
            required={false}
          >
            <Form.List name="static_headers">
              {(fields, { add, remove }) => (
                <div className="space-y-3">
                  {fields.map(({ key, name, ...restField }) => (
                    <Space key={key} className="flex w-full" align="baseline" size="middle">
                      <Form.Item
                        {...restField}
                        name={[name, "header"]}
                        className="flex-1"
                        rules={[{ required: true, message: "Header name is required" }]}
                      >
                        <Input
                          size="large"
                          allowClear
                          className="rounded-lg"
                          placeholder="Header name (e.g., X-API-Key)"
                        />
                      </Form.Item>
                      <Form.Item
                        {...restField}
                        name={[name, "value"]}
                        className="flex-1"
                        rules={[{ required: true, message: "Header value is required" }]}
                      >
                        <Input
                          size="large"
                          allowClear
                          className="rounded-lg"
                          placeholder="Header value"
                        />
                      </Form.Item>
                      <MinusCircleOutlined
                        onClick={() => remove(name)}
                        className="text-gray-500 hover:text-red-500 cursor-pointer"
                      />
                    </Space>
                  ))}
                  <Button type="dashed" onClick={() => add()} icon={<PlusOutlined />} block>
                    Add Static Header
                  </Button>
                </div>
              )}
            </Form.List>
          </Form.Item>
        </div>
      </Panel>
    </Collapse>
  );
};

export default MCPPermissionManagement;
