import React from "react";
import { Text } from "@tremor/react";
import { MCPServerCostInfo, toFiniteNumber } from "./types";

interface MCPServerCostDisplayProps {
  costConfig?: MCPServerCostInfo | null;
}

const MCPServerCostDisplay: React.FC<MCPServerCostDisplayProps> = ({ costConfig }) => {
  // Coerce eagerly: bad data (e.g. YAML-1.1 strings round-tripped via the DB)
  // would otherwise crash on .toFixed(). See backend issue #27097.
  const defaultCost = toFiniteNumber(costConfig?.default_cost_per_query);
  const toolCostEntries = Object.entries(costConfig?.tool_name_to_cost_per_query ?? {})
    .map(([toolName, raw]) => [toolName, toFiniteNumber(raw)] as const)
    .filter(([, n]) => n !== null) as Array<readonly [string, number]>;

  const hasDefaultCost = defaultCost !== null;
  const hasToolCosts = toolCostEntries.length > 0;
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
              {toolCostEntries.map(([toolName, cost]) => (
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
              <Text className="text-blue-700">
                • Default cost: ${defaultCost.toFixed(4)} per query
              </Text>
            )}
            {hasToolCosts && (
              <Text className="text-blue-700">
                • {toolCostEntries.length} tool(s) with custom pricing
              </Text>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MCPServerCostDisplay;
