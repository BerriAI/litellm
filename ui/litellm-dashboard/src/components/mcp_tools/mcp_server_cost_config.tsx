import React, { useState, useEffect } from "react";
import { Tooltip, InputNumber, Button, message, Spin, Alert } from "antd";
import { InfoCircleOutlined, DollarOutlined, ReloadOutlined } from "@ant-design/icons";
import { Card, Title, Text } from "@tremor/react";
import { MCPServerCostInfo } from "./types";
import { testMCPToolsListRequest } from "../networking";

interface MCPServerCostConfigProps {
  value?: MCPServerCostInfo;
  onChange?: (value: MCPServerCostInfo) => void;
  serverId?: string;
  serverUrl?: string;
  accessToken: string | null;
  disabled?: boolean;
  formValues?: Record<string, any>; // Add form values to fetch tools
}

const MCPServerCostConfig: React.FC<MCPServerCostConfigProps> = ({
  value = {},
  onChange,
  serverId,
  serverUrl,
  accessToken,
  disabled = false,
  formValues = {}
}) => {
  const [tools, setTools] = useState<any[]>([]);
  const [isLoadingTools, setIsLoadingTools] = useState(false);
  const [toolsError, setToolsError] = useState<string | null>(null);

  const handleDefaultCostChange = (defaultCost: number | null) => {
    const updated = {
      ...value,
      default_cost_per_query: defaultCost
    };
    onChange?.(updated);
  };

  const handleToolCostChange = (toolName: string, cost: number | null) => {
    const updated = {
      ...value,
      tool_costs: {
        ...value.tool_name_to_cost_per_query,
        [toolName]: cost
      }
    };
    onChange?.(updated);
  };

  const fetchTools = async () => {
    if (!accessToken || !formValues.url) {
      return;
    }

    setIsLoadingTools(true);
    setToolsError(null);
    
    try {
      // Prepare the MCP server config from form values
      const mcpServerConfig = {
        server_id: formValues.server_id || "",
        alias: formValues.alias || "",
        url: formValues.url,
        transport: formValues.transport,
        spec_version: formValues.spec_version,
        auth_type: formValues.auth_type,
        mcp_info: formValues.mcp_info,
      };

      const toolsResponse = await testMCPToolsListRequest(accessToken, mcpServerConfig);
      
      if (toolsResponse.tools && !toolsResponse.error) {
        setTools(toolsResponse.tools);
        setToolsError(null);
      } else {
        const errorMessage = toolsResponse.message || "Failed to retrieve tools list";
        setToolsError(errorMessage);
        setTools([]);
      }
    } catch (error) {
      console.error("Tools fetch error:", error);
      setToolsError(error instanceof Error ? error.message : String(error));
      setTools([]);
    } finally {
      setIsLoadingTools(false);
    }
  };

  // Auto-fetch tools when form values change and URL is available
  useEffect(() => {
    if (formValues.url && accessToken) {
      fetchTools();
    }
  }, [formValues.url, formValues.transport, formValues.auth_type, accessToken]);

  return (
    <Card>
      <div className="space-y-6">
        <div className="flex items-center gap-2 mb-4">
          <DollarOutlined className="text-green-600" />
          <Title>Cost Configuration</Title>
          <Tooltip title="Configure costs for this MCP server's tool calls. These costs will be tracked when the server's tools are used.">
            <InfoCircleOutlined className="text-gray-400" />
          </Tooltip>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Default Cost per Query ($)
              <Tooltip title="Default cost charged for each tool call to this server.">
                <InfoCircleOutlined className="ml-1 text-gray-400" />
              </Tooltip>
            </label>
            <InputNumber
              min={0}
              step={0.0001}
              precision={4}
              placeholder="0.0000"
              value={value.default_cost_per_query}
              onChange={handleDefaultCostChange}
              disabled={disabled}
              style={{ width: '200px' }}
              addonBefore="$"
            />
            <Text className="block mt-1 text-gray-500 text-sm">
              Set a default cost for all tool calls to this server
            </Text>
          </div>
        </div>

          {/* Tools-specific costs section */}
          {formValues.url && (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="block text-sm font-medium text-gray-700">
                    Tool-Specific Costs ($)
                    <Tooltip title="Set specific costs for individual tools. If not set, the default cost will be used.">
                      <InfoCircleOutlined className="ml-1 text-gray-400" />
                    </Tooltip>
                  </label>
                  <Text className="text-gray-500 text-sm">
                    Configure costs for specific tools from this MCP server
                  </Text>
                </div>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={fetchTools}
                  loading={isLoadingTools}
                  disabled={disabled}
                  size="small"
                >
                  Refresh Tools
                </Button>
              </div>

              {isLoadingTools && (
                <div className="flex items-center justify-center py-8">
                  <Spin size="large" />
                  <Text className="ml-3">Loading available tools...</Text>
                </div>
              )}

              {toolsError && (
                <Alert
                  message="Error loading tools"
                  description={toolsError}
                  type="warning"
                  showIcon
                  className="mb-4"
                />
              )}

              {!isLoadingTools && tools.length > 0 && (
                <div className="space-y-3 max-h-64 overflow-y-auto border rounded-lg p-4">
                  {tools.map((tool, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex-1">
                        <Text className="font-medium text-gray-900">{tool.name}</Text>
                        {tool.description && (
                          <Text className="text-gray-500 text-sm block mt-1">
                            {tool.description}
                          </Text>
                        )}
                      </div>
                      <div className="ml-4">
                        <InputNumber
                          min={0}
                          step={0.0001}
                          precision={4}
                          placeholder="Use default"
                          value={value.tool_name_to_cost_per_query?.[tool.name]}
                          onChange={(cost) => handleToolCostChange(tool.name, cost)}
                          disabled={disabled}
                          style={{ width: '120px' }}
                          addonBefore="$"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {!isLoadingTools && tools.length === 0 && !toolsError && formValues.url && (
                <div className="text-center py-8 text-gray-500">
                  <Text>No tools found for this MCP server</Text>
                </div>
              )}
            </div>
          )}

        {(value.default_cost_per_query || (value.tool_costs && Object.keys(value.tool_costs).length > 0)) && (
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <Text className="text-blue-800 font-medium">Cost Summary:</Text>
            <div className="mt-2 space-y-1">
              {value.default_cost_per_query && (
                <Text className="text-blue-700">
                  • Default cost: ${value.default_cost_per_query.toFixed(4)} per query
                </Text>
              )}
              {value.tool_name_to_cost_per_query && Object.entries(value.tool_name_to_cost_per_query).map(([toolName, cost]) => 
                cost !== null && cost !== undefined && (
                  <Text key={toolName} className="text-blue-700">
                    • {toolName}: ${cost.toFixed(4)} per query
                  </Text>
                )
              )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPServerCostConfig; 