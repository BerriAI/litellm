import React, { useState } from "react";
import { Form, Button as AntdButton, message, Collapse, Tooltip, Table, InputNumber } from "antd";
import { InfoCircleOutlined, DollarOutlined, ToolOutlined } from "@ant-design/icons";
import { Button, TextInput, Card, Title, Text } from "@tremor/react";
import { MCPServerCostInfo, MCPTool } from "./types";
import { listMCPTools } from "../networking";

interface MCPServerCostConfigProps {
  value?: MCPServerCostInfo;
  onChange?: (value: MCPServerCostInfo) => void;
  serverId?: string;
  serverUrl?: string;
  accessToken: string | null;
  disabled?: boolean;
}

const MCPServerCostConfig: React.FC<MCPServerCostConfigProps> = ({
  value = {},
  onChange,
  serverId,
  serverUrl,
  accessToken,
  disabled = false
}) => {
  const [loading, setLoading] = useState(false);
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [toolsLoaded, setToolsLoaded] = useState(false);
  
  const handleDefaultCostChange = (defaultCost: number | null) => {
    const updated = {
      ...value,
      default_cost_per_query: defaultCost
    };
    onChange?.(updated);
  };

  const handleToolCostChange = (toolName: string, cost: number | null) => {
    const toolCosts = { ...value.tool_name_to_cost_per_query };
    
    if (cost === null) {
      delete toolCosts[toolName];
    } else {
      toolCosts[toolName] = cost;
    }

    const updated = {
      ...value,
      tool_name_to_cost_per_query: Object.keys(toolCosts).length > 0 ? toolCosts : null
    };
    onChange?.(updated);
  };

  const fetchServerTools = async () => {
    if (!serverId || !accessToken) {
      message.error("Server ID and access token required to fetch tools");
      return;
    }

    setLoading(true);
    try {
      const response = await listMCPTools(accessToken, serverId);
      
      if (response.error) {
        message.error(`Failed to fetch tools: ${response.message}`);
        return;
      }

      if (response.tools && Array.isArray(response.tools)) {
        setTools(response.tools);
        setToolsLoaded(true);
        message.success(`Loaded ${response.tools.length} tools from server`);
      } else {
        message.warning("No tools found on this server");
        setTools([]);
        setToolsLoaded(true);
      }
    } catch (error) {
      console.error("Error fetching tools:", error);
      message.error("Failed to fetch tools from server");
    } finally {
      setLoading(false);
    }
  };

  const toolTableColumns = [
    {
      title: 'Tool Name',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => (
        <div className="flex items-center gap-2">
          <ToolOutlined className="text-blue-500" />
          <Text className="font-medium">{name}</Text>
        </div>
      ),
    },
    {
      title: 'Description',
      dataIndex: 'description',
      key: 'description',
      render: (description: string) => (
        <Text className="text-gray-600">{description || 'No description'}</Text>
      ),
    },
    {
      title: 'Cost per Query ($)',
      key: 'cost',
      render: (tool: MCPTool) => (
        <InputNumber
          min={0}
          step={0.0001}
          precision={4}
          placeholder="Use default"
          value={value.tool_name_to_cost_per_query?.[tool.name] || null}
          onChange={(cost) => handleToolCostChange(tool.name, cost)}
          disabled={disabled}
          style={{ width: '120px' }}
        />
      ),
    },
  ];

  const collapseItems = [
    {
      key: 'granular-costs',
      label: (
        <div className="flex items-center gap-2">
          <ToolOutlined />
          <span>Granular Tool Costs</span>
          <Tooltip title="Set custom costs for individual tools. Tools without custom costs will use the default cost.">
            <InfoCircleOutlined className="text-gray-400" />
          </Tooltip>
        </div>
      ),
      children: (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Text>Configure custom costs for individual tools on this server.</Text>
            <Button
              onClick={fetchServerTools}
              loading={loading}
              disabled={disabled || !serverId}
              size="sm"
              variant="secondary"
            >
              {toolsLoaded ? 'Refresh Tools' : 'Load Tools'}
            </Button>
          </div>
          
          {!serverId && (
            <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
              <Text className="text-yellow-800">
                Save the server first to load tools for granular cost configuration.
              </Text>
            </div>
          )}

          {toolsLoaded && (
            <Table
              dataSource={tools}
              columns={toolTableColumns}
              rowKey="name"
              size="small"
              pagination={false}
              locale={{ emptyText: 'No tools found on this server' }}
            />
          )}
        </div>
      ),
    },
  ];

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
              <Tooltip title="Default cost charged for each tool call to this server. Used when no specific tool cost is set.">
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

          <Collapse 
            items={collapseItems}
            ghost
            className="bg-gray-50"
          />
        </div>

        {(value.default_cost_per_query || value.tool_name_to_cost_per_query) && (
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <Text className="text-blue-800 font-medium">Cost Summary:</Text>
            <div className="mt-2 space-y-1">
              {value.default_cost_per_query && (
                <Text className="text-blue-700">
                  • Default cost: ${value.default_cost_per_query.toFixed(4)} per query
                </Text>
              )}
              {value.tool_name_to_cost_per_query && Object.keys(value.tool_name_to_cost_per_query).length > 0 && (
                <Text className="text-blue-700">
                  • Custom costs set for {Object.keys(value.tool_name_to_cost_per_query).length} tool(s)
                </Text>
              )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPServerCostConfig; 