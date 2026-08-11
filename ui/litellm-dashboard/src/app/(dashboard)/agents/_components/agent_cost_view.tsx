import React from "react";
import { Agent } from "@/components/agents/types";
import { useTranslation } from "react-i18next";

interface AgentCostViewProps {
  agent: Agent;
}

const AgentCostView: React.FC<AgentCostViewProps> = ({ agent }) => {
  const { t } = useTranslation("gateway");
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
      [t("agents.costs.perQuery"), params.cost_per_query],
      [t("agents.costs.inputToken"), params.input_cost_per_token],
      [t("agents.costs.outputToken"), params.output_cost_per_token],
    ] as const
  ).filter(([, value]) => value !== undefined);

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold text-foreground">{t("agents.costs.title")}</h3>
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
