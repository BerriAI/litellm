import React from "react";
import { useTranslation } from "react-i18next";
import { Text } from "@tremor/react";
import { MCPServerCostInfo } from "./types";

interface MCPServerCostDisplayProps {
  costConfig?: MCPServerCostInfo | null;
}

const MCPServerCostDisplay: React.FC<MCPServerCostDisplayProps> = ({ costConfig }) => {
  const { t } = useTranslation();
  const hasDefaultCost =
    costConfig?.default_cost_per_query !== undefined && costConfig?.default_cost_per_query !== null;
  const hasToolCosts =
    costConfig?.tool_name_to_cost_per_query && Object.keys(costConfig.tool_name_to_cost_per_query).length > 0;
  const hasCostConfig = hasDefaultCost || hasToolCosts;

  if (!hasCostConfig) {
    return (
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="space-y-4">
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <Text className="text-gray-600">{t("mcpTools.mcpServerCostDisplay.noCostConfig")}</Text>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6 pt-6 border-t border-gray-200">
      <div className="space-y-4">
        {hasDefaultCost &&
          costConfig?.default_cost_per_query !== undefined &&
          costConfig?.default_cost_per_query !== null && (
            <div>
              <Text className="font-medium">{t("mcpTools.mcpServerCostDisplay.defaultCostPerQuery")}</Text>
              <div className="text-green-600 font-mono">${costConfig.default_cost_per_query.toFixed(4)}</div>
            </div>
          )}

        {hasToolCosts && costConfig?.tool_name_to_cost_per_query && (
          <div>
            <Text className="font-medium">{t("mcpTools.mcpServerCostDisplay.toolSpecificCosts")}</Text>
            <div className="mt-2 space-y-2">
              {Object.entries(costConfig.tool_name_to_cost_per_query).map(
                ([toolName, cost]) =>
                  cost !== null &&
                  cost !== undefined && (
                    <div key={toolName} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <Text className="font-medium">{toolName}</Text>
                      <Text className="text-green-600 font-mono">
                        {t("mcpTools.mcpServerCostDisplay.costPerQuery", { cost: cost.toFixed(4) })}
                      </Text>
                    </div>
                  ),
              )}
            </div>
          </div>
        )}

        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <Text className="text-blue-800 font-medium">{t("mcpTools.mcpServerCostDisplay.costSummary")}</Text>
          <div className="mt-2 space-y-1">
            {hasDefaultCost &&
              costConfig?.default_cost_per_query !== undefined &&
              costConfig?.default_cost_per_query !== null && (
                <Text className="text-blue-700">
                  {t("mcpTools.mcpServerCostDisplay.defaultCostRow", {
                    cost: costConfig.default_cost_per_query.toFixed(4),
                  })}
                </Text>
              )}
            {hasToolCosts && costConfig?.tool_name_to_cost_per_query && (
              <Text className="text-blue-700">
                {t("mcpTools.mcpServerCostDisplay.toolsWithCustomPricing", {
                  count: Object.keys(costConfig.tool_name_to_cost_per_query).length,
                })}
              </Text>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MCPServerCostDisplay;
