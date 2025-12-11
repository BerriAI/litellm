import React from "react";
import { Form, Input, Select } from "antd";
import { AgentCreateInfo, AgentCredentialFieldMetadata } from "../networking";

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
        label="Display Name"
        name="display_name"
        tooltip="Human-readable name shown in the UI"
      >
        <Input placeholder="e.g., My LangGraph Agent" />
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
    </>
  );
};

/**
 * Builds agent data from form values for dynamic agent types.
 */
export const buildDynamicAgentData = (
  values: any,
  agentTypeInfo: AgentCreateInfo
) => {
  return {
    agent_name: values.agent_name,
    agent_card_params: {
      protocolVersion: "1.0",
      name: values.display_name || values.agent_name,
      description: values.description || `${agentTypeInfo.agent_type_display_name} agent`,
      url: values.api_base,
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
    litellm_params: {
      ...(agentTypeInfo.litellm_params_template || {}),
      ...(values.api_key && { api_key: values.api_key }),
      ...(values.model && { model: values.model }),
    },
  };
};

export default DynamicAgentFormFields;

