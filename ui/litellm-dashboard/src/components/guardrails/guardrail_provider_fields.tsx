import React from "react";
import { Form, Select } from "antd";
import { TextInput } from "@tremor/react";
import { GuardrailProviders } from './guardrail_info_helpers';

interface GuardrailProviderFieldsProps {
  selectedProvider: string | null;
}

interface ProviderField {
  key: string;
  label: string;
  placeholder?: string;
  tooltip?: string;
  required?: boolean;
  type?: "text" | "password" | "select";
  options?: string[];
  defaultValue?: string;
}

// Define fields for each guardrail provider
const GUARDRAIL_PROVIDER_FIELDS: Record<string, ProviderField[]> = {
  // Presidio PII fields
  PresidioPII: [
    {
      key: "presidio_analyzer_api_base",
      label: "Presidio Analyzer API Base",
      placeholder: "https://your-analyzer-api-url",
      tooltip: "The base URL for your Presidio Analyzer API",
      required: true
    },
    {
      key: "presidio_anonymizer_api_base",
      label: "Presidio Anonymizer API Base",
      placeholder: "https://your-anonymizer-api-url",
      tooltip: "The base URL for your Presidio Anonymizer API",
      required: true
    }
  ],
  // Add more provider specific fields here as needed
  Bedrock: [
    {
      key: "config",
      label: "Configuration",
      placeholder: '{"guardrail_id": "...", "guardrail_version": "..."}',
      tooltip: "JSON configuration for Bedrock guardrail including guardrail_id and guardrail_version",
      type: "text",
      required: false
    }
  ]
};

const GuardrailProviderFields: React.FC<GuardrailProviderFieldsProps> = ({
  selectedProvider
}) => {
  // If no provider is selected or if the provider doesn't have specific fields, return null
  if (!selectedProvider || !GUARDRAIL_PROVIDER_FIELDS[selectedProvider]) {
    return null;
  }

  const providerFields = GUARDRAIL_PROVIDER_FIELDS[selectedProvider];

  return (
    <>
      {providerFields.map((field) => (
        <Form.Item
          key={field.key}
          name={field.key}
          label={field.label}
          tooltip={field.tooltip}
          rules={field.required ? [{ required: true, message: `Please enter ${field.label}` }] : undefined}
        >
          {field.type === "select" ? (
            <Select placeholder={field.placeholder} defaultValue={field.defaultValue}>
              {field.options?.map((option) => (
                <Select.Option key={option} value={option}>
                  {option}
                </Select.Option>
              ))}
            </Select>
          ) : (
            <TextInput
              placeholder={field.placeholder}
              type={field.type === "password" ? "password" : "text"}
            />
          )}
        </Form.Item>
      ))}
    </>
  );
};

export default GuardrailProviderFields; 