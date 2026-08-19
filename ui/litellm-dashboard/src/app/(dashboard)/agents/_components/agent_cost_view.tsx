import React from "react";
import { Agent } from "@/components/agents/types";

interface AgentCostViewProps {
  agent: Agent;
}

const AgentCostView: React.FC<AgentCostViewProps> = ({ agent }) => {
  const params = agent.litellm_params;

  if (
    params?.cost_per_query === undefined &&
    params?.input_cost_per_token === undefined &&
    params?.output_cost_per_token === undefined
  ) {
    return null;
  }

  const rows = (
    [
      ["Cost Per Query", params.cost_per_query],
      ["Input Cost Per Token", params.input_cost_per_token],
      ["Output Cost Per Token", params.output_cost_per_token],
    ] as const
  ).filter(([, value]) => value !== undefined);

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold text-foreground">Cost Configuration</h3>
      <dl className="mt-4 divide-y divide-border overflow-hidden rounded-lg border border-border">
        {rows.map(([label, value]) => (
          <div key={label} className="grid grid-cols-1 sm:grid-cols-3">
            <dt className="bg-muted/50 px-4 py-3 text-sm font-medium text-foreground">{label}</dt>
            <dd className="px-4 py-3 text-sm text-foreground sm:col-span-2">${value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
};

export default AgentCostView;
