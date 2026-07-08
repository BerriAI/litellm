import React from "react";
import { Text } from "@tremor/react";
import { MCPServerCostInfo } from "./types";
import { toFiniteNumber } from "../../utils/numberUtils";

interface MCPServerCostDisplayProps {
  costConfig?: MCPServerCostInfo | null;
}

const MCPServerCostDisplay: React.FC<MCPServerCostDisplayProps> = ({ costConfig }) => {
  // Cost values arrive from JSONB / YAML / antd InputNumber and may be
  // stringified numbers — coerce defensively so `.toFixed(...)` is safe.
  const defaultCost = toFiniteNumber(costConfig?.default_cost_per_query);
  const toolCosts: Array<[string, number]> = Object.entries(costConfig?.tool_name_to_cost_per_query ?? {})
    .map(([name, raw]): [string, number | null] => [name, toFiniteNumber(raw)])
    .filter((entry): entry is [string, number] => entry[1] !== null);

  const hasDefaultCost = defaultCost !== null;
  const hasToolCosts = toolCosts.length > 0;
  const hasCostConfig = hasDefaultCost || hasToolCosts;

  if (!hasCostConfig) {
    return (
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="space-y-4">
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <Text className="text-gray-600">
              No cost configuration set for this server. Tool calls will be charged at $0.00 per tool call.
            </Text>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6 pt-6 border-t border-gray-200">
      <div className="space-y-4">
        {hasDefaultCost && defaultCost !== null && (
          <div>
            <Text className="font-medium">Default Cost per Query</Text>
            <div className="text-green-600 font-mono">${defaultCost.toFixed(4)}</div>
          </div>
        )}

        {hasToolCosts && (
          <div>
            <Text className="font-medium">Tool-Specific Costs</Text>
            <div className="mt-2 space-y-2">
              {toolCosts.map(([toolName, cost]) => (
                <div key={toolName} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <Text className="font-medium">{toolName}</Text>
                  <Text className="text-green-600 font-mono">${cost.toFixed(4)} per query</Text>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <Text className="text-blue-800 font-medium">Cost Summary:</Text>
          <div className="mt-2 space-y-1">
            {hasDefaultCost && defaultCost !== null && (
              <Text className="text-blue-700">• Default cost: ${defaultCost.toFixed(4)} per query</Text>
            )}
            {hasToolCosts && (
              <Text className="text-blue-700">• {toolCosts.length} tool(s) with custom pricing</Text>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MCPServerCostDisplay;
