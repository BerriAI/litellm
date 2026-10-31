import React from "react";
import { useTranslation } from "react-i18next";
import { Agent } from "@/components/agents/types";

interface AgentCostViewProps {
  agent: Agent;
}

const AgentCostView: React.FC<AgentCostViewProps> = ({ agent }) => {
  const { t } = useTranslation();
  const params = agent.litellm_params;

  if (params?.cost_per_query == null && params?.input_cost_per_token == null && params?.output_cost_per_token == null) {
    return null;
  }

  const rows = (
    [
      [t("agents.costView.costPerQuery"), params.cost_per_query],
      [t("agents.costView.inputCostPerToken"), params.input_cost_per_token],
      [t("agents.costView.outputCostPerToken"), params.output_cost_per_token],
    ] as const
  ).filter(([, value]) => value != null);

  return (
    <div className="mt-6">
      <h3 className="text-lg font-semibold text-foreground">{t("agents.costView.title")}</h3>
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
