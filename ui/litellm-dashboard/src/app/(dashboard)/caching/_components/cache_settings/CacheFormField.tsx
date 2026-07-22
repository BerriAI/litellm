import { Form, Input, Select, Switch } from "antd";
import React from "react";
import { CacheField } from "./cacheSettingsFields";

export interface EmbeddingModelOption {
  value: string;
  label: string;
}

export const SECRET_ALREADY_SET_PLACEHOLDER = "Already set. Enter a new value to replace it.";

interface CacheFormFieldProps {
  field: CacheField;
  embeddingModels: EmbeddingModelOption[];
  isSecretConfigured?: boolean;
}

const renderControl = (
  field: CacheField,
  embeddingModels: EmbeddingModelOption[],
  placeholder: string,
): React.ReactNode => {
  switch (field.type) {
    case "boolean":
      return <Switch />;
    case "password":
      return <Input.Password placeholder={placeholder} autoComplete="new-password" />;
    case "integer":
    case "float":
      return <Input inputMode="decimal" placeholder={placeholder} />;
    case "list":
      return <Input.TextArea rows={4} placeholder={placeholder} />;
    case "model-select":
      return (
        <Select
          showSearch
          allowClear
          placeholder="Search and select a model..."
          options={embeddingModels}
          optionFilterProp="label"
          style={{ width: "100%" }}
        />
      );
    default:
      return <Input placeholder={placeholder} />;
  }
};

const CacheFormField: React.FC<CacheFormFieldProps> = ({ field, embeddingModels, isSecretConfigured = false }) => (
  <Form.Item
    name={field.name}
    label={field.label}
    extra={field.helpText}
    rules={field.rules}
    valuePropName={field.type === "boolean" ? "checked" : "value"}
  >
    {renderControl(field, embeddingModels, isSecretConfigured ? SECRET_ALREADY_SET_PLACEHOLDER : field.helpText)}
  </Form.Item>
);

export default CacheFormField;
