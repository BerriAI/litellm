import React, { useEffect } from "react";
import { Form, Select, Tooltip, Collapse } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
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
        </div>
      </Panel>
    </Collapse>
  );
};

export default MCPPermissionManagement;
