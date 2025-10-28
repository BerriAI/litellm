import React from "react";
import { Form } from "antd";
import { TextInput } from "@tremor/react";
interface Field {
  field_name: string;
  field_type: string;
  field_description: string;
  field_value: string;
}

interface DynamicFieldsProps {
  fields: Field[];
  selectedProvider: string;
}

const getPlaceholder = (provider: string) => {
  // Implement your placeholder logic based on the provider
  return `Enter your ${provider} value here`;
};

const DynamicFields: React.FC<DynamicFieldsProps> = ({ fields, selectedProvider }) => {
  if (fields.length === 0) return null;

  return (
    <>
      {fields.map((field) => (
        <Form.Item
          key={field.field_name}
          rules={[{ required: true, message: "Required" }]}
          label={field.field_name.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase())}
          name={field.field_name}
          tooltip={field.field_description}
          className="mb-2"
        >
          <TextInput placeholder={field.field_value} type="password" />
        </Form.Item>
      ))}
    </>
  );
};

export default DynamicFields;
