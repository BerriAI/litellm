import React from "react";
import { Form, Input, InputNumber, Select as AntSelect, Switch } from "antd";
import type { FormInstance } from "antd";

interface CustomerFormFieldsProps {
  form: FormInstance;
  mode: "create" | "edit";
  disabledFields?: string[];
}

const defaultModelOptions = [
  { value: "", label: "None" },
  { value: "gpt-4o", label: "gpt-4o" },
  { value: "gpt-4o-mini", label: "gpt-4o-mini" },
  { value: "gpt-4-turbo", label: "gpt-4-turbo" },
  { value: "claude-3-sonnet", label: "claude-3-sonnet" },
  { value: "claude-3-opus", label: "claude-3-opus" },
  { value: "claude-3-haiku", label: "claude-3-haiku" },
];

const regionOptions = [
  { value: "", label: "Any region" },
  { value: "us", label: "US" },
  { value: "eu", label: "EU" },
];

const CustomerFormFields: React.FC<CustomerFormFieldsProps> = ({
  form,
  mode,
  disabledFields = [],
}) => {
  const isDisabled = (field: string) => disabledFields.includes(field);

  return (
    <>
      <Form.Item label="Alias" name="alias">
        <Input placeholder={mode === "create" ? "e.g. Acme Corp" : "Customer alias"} />
      </Form.Item>

      <div className="grid grid-cols-2 gap-4">
        <Form.Item label="Max Budget" name="max_budget">
          <InputNumber
            className="w-full"
            placeholder={mode === "create" ? "e.g. 500" : "No limit"}
            min={0}
            disabled={isDisabled("max_budget")}
          />
        </Form.Item>
        <Form.Item label="Budget ID" name="budget_id">
          <Input placeholder="e.g. free_tier" />
        </Form.Item>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Form.Item label="Default Model" name="default_model">
          <AntSelect
            placeholder={mode === "create" ? "Select model..." : "None"}
            allowClear={mode === "edit"}
            options={defaultModelOptions}
          />
        </Form.Item>
        <Form.Item label="Allowed Region" name="allowed_model_region">
          <AntSelect
            placeholder="Any region"
            allowClear={mode === "edit"}
            options={regionOptions}
          />
        </Form.Item>
      </div>

      <Form.Item label="Budget Duration" name="budget_duration">
        <Input placeholder={mode === "create" ? "e.g. 30d, 24h, 60m" : "e.g. 30d, 24h"} />
      </Form.Item>

      {mode === "edit" && (
        <Form.Item label="Blocked" name="blocked" valuePropName="checked">
          <div className="flex items-center gap-3">
            <Switch />
            <span className="text-sm text-gray-500">
              {form.getFieldValue("blocked")
                ? "This customer is currently blocked from making requests"
                : "This customer can make requests normally"}
            </span>
          </div>
        </Form.Item>
      )}
    </>
  );
};

export default CustomerFormFields;
