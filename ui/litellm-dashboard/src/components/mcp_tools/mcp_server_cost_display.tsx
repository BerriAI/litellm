import React from "react";
import { MCPServerCostInfo } from "./types";

interface MCPServerCostDisplayProps {
  costConfig?: MCPServerCostInfo | null;
}

const MCPServerCostDisplay: React.FC<MCPServerCostDisplayProps> = ({
  costConfig,
}) => {
  const hasDefaultCost =
    costConfig?.default_cost_per_query !== undefined &&
    costConfig?.default_cost_per_query !== null;
  const hasToolCosts =
    costConfig?.tool_name_to_cost_per_query &&
    Object.keys(costConfig.tool_name_to_cost_per_query).length > 0;
  const hasCostConfig = hasDefaultCost || hasToolCosts;

  if (!hasCostConfig) {
    return (
      <div className="mt-6 pt-6 border-t border-border">
        <div className="space-y-4">
          <div className="p-4 bg-muted border border-border rounded-lg">
            <p className="text-muted-foreground">
              No cost configuration set for this server. Tool calls will be
              charged at $0.00 per tool call.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6 pt-6 border-t border-border">
      <div className="space-y-4">
        {hasDefaultCost &&
          costConfig?.default_cost_per_query !== undefined &&
          costConfig?.default_cost_per_query !== null && (
            <div>
              <p className="font-medium">Default Cost per Query</p>
              <div className="text-emerald-600 dark:text-emerald-400 font-mono">
                ${costConfig.default_cost_per_query.toFixed(4)}
              </div>
            </div>
          )}

        {hasToolCosts && costConfig?.tool_name_to_cost_per_query && (
          <div>
            <p className="font-medium">Tool-Specific Costs</p>
            <div className="mt-2 space-y-2">
              {Object.entries(costConfig.tool_name_to_cost_per_query).map(
                ([toolName, cost]) =>
                  cost !== null &&
                  cost !== undefined && (
                    <div
                      key={toolName}
                      className="flex justify-between items-center p-3 bg-muted rounded-lg"
                    >
                      <span className="font-medium">{toolName}</span>
                      <span className="text-emerald-600 dark:text-emerald-400 font-mono">
                        ${cost.toFixed(4)} per query
                      </span>
                    </div>
                  ),
              )}
            </div>
          </div>
        )}

        <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg">
          <p className="text-blue-800 dark:text-blue-200 font-medium">
            Cost Summary:
          </p>
          <div className="mt-2 space-y-1">
            {hasDefaultCost &&
              costConfig?.default_cost_per_query !== undefined &&
              costConfig?.default_cost_per_query !== null && (
                <p className="text-blue-700 dark:text-blue-300">
                  • Default cost: $
                  {costConfig.default_cost_per_query.toFixed(4)} per query
                </p>
              )}
            {hasToolCosts && costConfig?.tool_name_to_cost_per_query && (
              <p className="text-blue-700 dark:text-blue-300">
                • {Object.keys(costConfig.tool_name_to_cost_per_query).length}{" "}
                tool(s) with custom pricing
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MCPServerCostDisplay;
