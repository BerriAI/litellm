import React from "react";
import { Form, Input } from "antd";
import { AGENT_FORM_CONFIG } from "./agent_config";
import { useTranslation } from "react-i18next";

const CostConfigFields: React.FC = () => {
  const { t } = useTranslation("gateway");
  const copy: Record<string, { label: string; tooltip: string }> = {
    cost_per_query: { label: t("agents.form.costPerQuery"), tooltip: t("agents.form.costPerQueryHint") },
    input_cost_per_token: { label: t("agents.form.inputCost"), tooltip: t("agents.form.inputCostHint") },
    output_cost_per_token: { label: t("agents.form.outputCost"), tooltip: t("agents.form.outputCostHint") },
  };
  return (
    <>
      {AGENT_FORM_CONFIG.cost.fields.map((field) => (
        <Form.Item
          key={field.name}
          label={copy[field.name]?.label ?? field.label}
          name={field.name}
          tooltip={copy[field.name]?.tooltip ?? field.tooltip}
        >
          <Input placeholder={field.placeholder} type="number" step="0.000001" />
        </Form.Item>
      ))}
    </>
  );
};

export default CostConfigFields;
