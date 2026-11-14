import React from "react";
import { useTranslation } from "react-i18next";
import { MCPServerCostInfo } from "@/components/mcp_tools/types";

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
      <div className="mt-6 border-t border-border pt-6">
        <div className="space-y-4">
          <div className="rounded-lg border border-border bg-muted p-4">
            <p className="text-sm text-muted-foreground">{t("mcpServers.serverCostDisplay.noConfig")}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6 border-t border-border pt-6">
      <div className="space-y-4">
        {hasDefaultCost &&
          costConfig?.default_cost_per_query !== undefined &&
          costConfig?.default_cost_per_query !== null && (
            <div>
              <p className="text-sm font-medium">{t("mcpServers.serverCostDisplay.defaultCostTitle")}</p>
              <div className="font-mono text-sm">${costConfig.default_cost_per_query.toFixed(4)}</div>
            </div>
          )}

        {hasToolCosts && costConfig?.tool_name_to_cost_per_query && (
          <div>
            <p className="text-sm font-medium">{t("mcpServers.serverCostDisplay.toolCostsTitle")}</p>
            <div className="mt-2 space-y-2">
              {Object.entries(costConfig.tool_name_to_cost_per_query).map(
                ([toolName, cost]) =>
                  cost !== null &&
                  cost !== undefined && (
                    <div key={toolName} className="flex items-center justify-between rounded-lg bg-muted p-3">
                      <p className="text-sm font-medium">{toolName}</p>
                      <p className="font-mono text-sm">
                        {t("mcpServers.serverCostDisplay.costPerQuery", { cost: cost.toFixed(4) })}
                      </p>
                    </div>
                  ),
              )}
            </div>
          </div>
        )}

        <div className="mt-4 rounded-lg border border-border bg-muted p-4">
          <p className="text-sm font-medium">{t("mcpServers.serverCostDisplay.summaryTitle")}</p>
          <div className="mt-2 space-y-1">
            {hasDefaultCost &&
              costConfig?.default_cost_per_query !== undefined &&
              costConfig?.default_cost_per_query !== null && (
                <p className="text-sm text-muted-foreground">
                  {t("mcpServers.serverCostDisplay.summaryDefaultCost", {
                    cost: costConfig.default_cost_per_query.toFixed(4),
                  })}
                </p>
              )}
            {hasToolCosts && costConfig?.tool_name_to_cost_per_query && (
              <p className="text-sm text-muted-foreground">
                {t("mcpServers.serverCostDisplay.summaryToolCount", {
                  toolCount: Object.keys(costConfig.tool_name_to_cost_per_query).length,
                })}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MCPServerCostDisplay;
