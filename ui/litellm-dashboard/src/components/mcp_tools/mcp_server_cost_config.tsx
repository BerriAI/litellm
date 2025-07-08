import React from "react";
import { Tooltip, InputNumber } from "antd";
import { InfoCircleOutlined, DollarOutlined } from "@ant-design/icons";
import { Card, Title, Text } from "@tremor/react";
import { MCPServerCostInfo } from "./types";

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
  const handleDefaultCostChange = (defaultCost: number | null) => {
    const updated = {
      ...value,
      default_cost_per_query: defaultCost
    };
    onChange?.(updated);
  };

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

        {value.default_cost_per_query && (
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <Text className="text-blue-800 font-medium">Cost Summary:</Text>
            <div className="mt-2 space-y-1">
              <Text className="text-blue-700">
                • Default cost: ${value.default_cost_per_query.toFixed(4)} per query
              </Text>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPServerCostConfig; 