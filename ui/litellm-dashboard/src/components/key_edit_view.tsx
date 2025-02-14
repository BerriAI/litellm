import React from "react";
import { Form, Input, InputNumber, Select } from "antd";
import { Button, TextInput } from "@tremor/react";
import { KeyResponse } from "./key_team_helpers/key_list";

interface KeyEditViewProps {
  keyData: KeyResponse;
  onCancel: () => void;
  onSubmit: (values: any) => Promise<void>;
}

export function KeyEditView({ keyData, onCancel, onSubmit }: KeyEditViewProps) {
  const [form] = Form.useForm();

  // Convert API budget duration to form format
  const getBudgetDuration = (duration: string | null) => {
    if (!duration) return null;
    const durationMap: Record<string, string> = {
      "24h": "daily",
      "7d": "weekly",
      "30d": "monthly"
    };
    return durationMap[duration] || null;
  };

  // Set initial form values
  const initialValues = {
    ...keyData,
    budget_duration: getBudgetDuration(keyData.budget_duration),
    metadata: keyData.metadata ? JSON.stringify(keyData.metadata, null, 2) : "",
    guardrails: keyData.metadata?.guardrails || []
  };

  return (
    <Form
      form={form}
      onFinish={onSubmit}
      initialValues={initialValues}
      layout="vertical"
    >
      <Form.Item label="Key Alias" name="key_alias">
        <TextInput />
      </Form.Item>

      <Form.Item label="Models" name="models">
        <Select
          mode="multiple"
          placeholder="Select models"
          style={{ width: "100%" }}
        >
          <Select.Option value="all-team-models">All Team Models</Select.Option>
          {/* Add model options based on team models */}
        </Select>
      </Form.Item>

      <Form.Item label="Max Budget (USD)" name="max_budget">
        <InputNumber step={0.01} precision={2} style={{ width: "100%" }} />
      </Form.Item>

      <Form.Item label="Reset Budget" name="budget_duration">
        <Select placeholder="n/a">
          <Select.Option value="daily">Daily</Select.Option>
          <Select.Option value="weekly">Weekly</Select.Option>
          <Select.Option value="monthly">Monthly</Select.Option>
        </Select>
      </Form.Item>

      <Form.Item label="TPM Limit" name="tpm_limit">
        <InputNumber style={{ width: "100%" }} />
      </Form.Item>

      <Form.Item label="RPM Limit" name="rpm_limit">
        <InputNumber style={{ width: "100%" }} />
      </Form.Item>

      <Form.Item label="Guardrails" name="guardrails">
        <Select
          mode="tags"
          style={{ width: "100%" }}
          placeholder="Select or enter guardrails"
        />
      </Form.Item>

      <Form.Item label="Metadata" name="metadata">
        <Input.TextArea rows={10} />
      </Form.Item>

      {/* Hidden form field for token */}
      <Form.Item name="token" hidden>
        <Input />
      </Form.Item>

      <div className="flex justify-end gap-2 mt-6">
        <Button variant="light" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="primary" htmlType="submit">
          Save Changes
        </Button>
      </div>
    </Form>
  );
} 