import React from "react";
import { Form, Input, Select, Collapse } from "antd";
import { AgentCreateInfo, AgentCredentialFieldMetadata } from "../networking";
import { AGENT_FORM_CONFIG } from "./agent_config";
import CostConfigFields from "./cost_config_fields";

const { Panel } = Collapse;

interface DynamicAgentFormFieldsProps {
  agentTypeInfo: AgentCreateInfo;
}

/**
 * Form fields for dynamic agent types (e.g., LangGraph).
 * Renders common fields (agent name, display name, description) plus
 * credential fields defined by the agent type metadata.
 */
const DynamicAgentFormFields: React.FC<DynamicAgentFormFieldsProps> = ({
  agentTypeInfo,
}) => {
  return (
    <>
      <Form.Item
        label="Agent Name"
        name="agent_name"
        rules={[{ required: true, message: "Please enter a unique agent name" }]}
        tooltip="Unique identifier for the agent"
      >
        <Input placeholder="e.g., my-langgraph-agent" />
      </Form.Item>

      <Form.Item
        label="Description"
        name="description"
        tooltip="Brief description of what this agent does"
      >
        <Input.TextArea rows={2} placeholder="Describe what this agent does..." />
      </Form.Item>

      {agentTypeInfo.credential_fields.map((field: AgentCredentialFieldMetadata) => (
        <Form.Item
          key={field.key}
          label={field.label}
          name={field.key}
          rules={field.required ? [{ required: true, message: `Please enter ${field.label}` }] : undefined}
          tooltip={field.tooltip}
          initialValue={field.default_value}
        >
          {field.field_type === "password" ? (
            <Input.Password placeholder={field.placeholder || ""} />
          ) : field.field_type === "textarea" ? (
            <Input.TextArea rows={3} placeholder={field.placeholder || ""} />
          ) : field.field_type === "select" && field.options ? (
            <Select placeholder={field.placeholder || ""}>
              {field.options.map((opt) => (
                <Select.Option key={opt} value={opt}>
                  {opt}
                </Select.Option>
              ))}
            </Select>
          ) : (
            <Input placeholder={field.placeholder || ""} />
          )}
        </Form.Item>
      ))}

      <Collapse style={{ marginBottom: 16 }}>
        <Panel header={AGENT_FORM_CONFIG.cost.title} key={AGENT_FORM_CONFIG.cost.key}>
          <CostConfigFields />
        </Panel>
      </Collapse>
    </>
  );
};

/**
 * Builds agent data from form values for dynamic agent types.
 * Uses configuration from agentTypeInfo to determine which fields to include.
 */
export const buildDynamicAgentData = (
  values: any,
  agentTypeInfo: AgentCreateInfo
) => {
  // Build litellm_params from template
  const litellmParams: Record<string, any> = {
    ...(agentTypeInfo.litellm_params_template || {}),
  };

  // Add credential fields marked with include_in_litellm_params
  for (const field of agentTypeInfo.credential_fields) {
    const value = values[field.key];
    if (value && field.include_in_litellm_params !== false) {
      litellmParams[field.key] = value;
    }
  }

  // Add cost configuration
  if (values.cost_per_query) {
    litellmParams.cost_per_query = parseFloat(values.cost_per_query);
  }
  if (values.input_cost_per_token) {
    litellmParams.input_cost_per_token = parseFloat(values.input_cost_per_token);
  }
  if (values.output_cost_per_token) {
    litellmParams.output_cost_per_token = parseFloat(values.output_cost_per_token);
  }

  // Apply model_template if defined (e.g., "bedrock/agentcore/{agent_runtime_arn}")
  if (agentTypeInfo.model_template) {
    let model = agentTypeInfo.model_template;
    // Replace {field_key} placeholders with actual values
    for (const field of agentTypeInfo.credential_fields) {
      const placeholder = `{${field.key}}`;
      if (model.includes(placeholder) && values[field.key]) {
        model = model.replace(placeholder, values[field.key]);
      }
    }
    litellmParams.model = model;
  }

  return {
    agent_name: values.agent_name,
    agent_card_params: {
      protocolVersion: "1.0",
      name: values.display_name || values.agent_name,
      description: values.description || `${agentTypeInfo.agent_type_display_name} agent`,
      url: values.api_base || "",
      version: "1.0.0",
      defaultInputModes: ["text"],
      defaultOutputModes: ["text"],
      capabilities: {
        streaming: true,
      },
      skills: [{
        id: "chat",
        name: "Chat",
        description: "General chat capability",
        tags: ["chat", "conversation"],
      }],
    },
    litellm_params: litellmParams,
  };
};

export default DynamicAgentFormFields;

