import React from "react";
import { Form, Input } from "antd";
import { useTranslation } from "react-i18next";
import { getAgentFormConfig } from "./agent_config";

const CostConfigFields: React.FC = () => {
  const { t } = useTranslation();
  return (
    <>
      {getAgentFormConfig(t).cost.fields.map((field) => (
        <Form.Item key={field.name} label={field.label} name={field.name} tooltip={field.tooltip}>
          <Input placeholder={field.placeholder} type="number" step="0.000001" />
        </Form.Item>
      ))}
    </>
  );
};

export default CostConfigFields;
