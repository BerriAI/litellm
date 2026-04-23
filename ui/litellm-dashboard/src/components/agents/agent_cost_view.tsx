import React from "react";
import { Agent } from "./types";

interface AgentCostViewProps {
  agent: Agent;
}

/**
 * Compact key-value panel for the cost-configuration fields on the agent
 * detail page.
 */
const AgentCostView: React.FC<AgentCostViewProps> = ({ agent }) => {
  const params = agent.litellm_params;

  if (
    params?.cost_per_query === undefined &&
    params?.input_cost_per_token === undefined &&
    params?.output_cost_per_token === undefined
  ) {
    return null;
  }

  const rows: Array<{ label: string; value: string }> = [];
  if (params.cost_per_query !== undefined)
    rows.push({
      label: "Cost Per Query",
      value: `$${params.cost_per_query}`,
    });
  if (params.input_cost_per_token !== undefined)
    rows.push({
      label: "Input Cost Per Token",
      value: `$${params.input_cost_per_token}`,
    });
  if (params.output_cost_per_token !== undefined)
    rows.push({
      label: "Output Cost Per Token",
      value: `$${params.output_cost_per_token}`,
    });

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold">Cost Configuration</h3>
      <div className="mt-4 border border-border rounded-md overflow-hidden">
        {rows.map((row, i) => (
          <div
            key={row.label}
            className={`grid grid-cols-[minmax(200px,1fr)_2fr] ${
              i !== 0 ? "border-t border-border" : ""
            }`}
          >
            <div className="bg-muted px-4 py-2 font-medium">{row.label}</div>
            <div className="px-4 py-2">{row.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentCostView;
