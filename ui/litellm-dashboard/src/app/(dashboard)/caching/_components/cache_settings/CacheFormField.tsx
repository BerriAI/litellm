import { Form, Input, Select, Switch } from "antd";
import React from "react";
import { CacheField } from "./cacheSettingsFields";

export interface EmbeddingModelOption {
  value: string;
  label: string;
}

interface CacheFormFieldProps {
  field: CacheField;
  embeddingModels: EmbeddingModelOption[];
}

const renderControl = (field: CacheField, embeddingModels: EmbeddingModelOption[]): React.ReactNode => {
  switch (field.type) {
    case "boolean":
      return <Switch />;
    case "password":
      return <Input.Password placeholder={field.helpText} autoComplete="new-password" />;
    case "integer":
    case "float":
      return <Input inputMode="decimal" placeholder={field.helpText} />;
    case "list":
      return <Input.TextArea rows={4} placeholder={field.helpText} />;
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
      return <Input placeholder={field.helpText} />;
  }
};

const CacheFormField: React.FC<CacheFormFieldProps> = ({ field, embeddingModels }) => (
  <Form.Item
    name={field.name}
    label={field.label}
    extra={field.helpText}
    rules={field.rules}
    valuePropName={field.type === "boolean" ? "checked" : "value"}
  >
    {renderControl(field, embeddingModels)}
  </Form.Item>
);

export default CacheFormField;
