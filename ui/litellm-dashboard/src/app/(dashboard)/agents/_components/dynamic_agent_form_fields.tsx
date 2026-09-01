import React from "react";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { FieldGroup } from "@/components/ui/field";
import { AgentCreateInfo, AgentCredentialFieldMetadata } from "@/components/networking";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { AGENT_FORM_CONFIG } from "./agent_config";
import CostConfigFields, { COST_FIELD_NAMES } from "./cost_config_fields";
import {
  AgentFormField,
  AgentFormPanel,
  AgentRequestPayload,
  AgentFormValues,
  CollapsiblePanelsState,
  labelWithHint,
} from "./AgentFormKit";

interface DynamicAgentFormFieldsProps {
  agentTypeInfo: AgentCreateInfo;
  panels: CollapsiblePanelsState;
}

export const unmountedDynamicFieldNames = (mountedPanels: readonly string[]): readonly string[] =>
  mountedPanels.includes(AGENT_FORM_CONFIG.cost.key) ? [] : COST_FIELD_NAMES;

const CredentialField = ({ field }: { field: AgentCredentialFieldMetadata }) => (
  <AgentFormField
    name={field.key}
    label={field.tooltip ? labelWithHint(field.label, field.tooltip) : field.label}
    defaultValue={field.default_value ?? undefined}
    rules={field.required ? { required: `Please enter ${field.label}` } : undefined}
  >
    {({ value, onChange, ref, ...control }) => {
      const text = typeof value === "string" ? value : "";
      if (field.field_type === "password") {
        return (
          <PasswordInput
            {...control}
            value={typeof value === "string" ? value : ""}
            onChange={onChange}
            ref={ref}
            placeholder={field.placeholder || ""}
          />
        );
      }
      if (field.field_type === "textarea") {
        return (
          <Textarea
            {...control}
            ref={ref}
            rows={3}
            placeholder={field.placeholder || ""}
            value={text}
            onChange={onChange}
          />
        );
      }
      if (field.field_type === "select" && field.options) {
        return (
          <Select value={text || null} onValueChange={onChange}>
            <SelectTrigger {...control} className="w-full">
              <SelectValue placeholder={field.placeholder || ""} />
            </SelectTrigger>
            <SelectContent>
              {field.options.map((option) => (
                <SelectItem key={option} value={option} title={option}>
                  {option}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );
      }
      return <Input {...control} ref={ref} placeholder={field.placeholder || ""} value={text} onChange={onChange} />;
    }}
  </AgentFormField>
);

const DynamicAgentFormFields: React.FC<DynamicAgentFormFieldsProps> = ({ agentTypeInfo, panels }) => (
  <>
    <FieldGroup className="mb-4">
      <AgentFormField
        name="agent_name"
        label={labelWithHint("Agent Name", "Unique identifier for the agent")}
        rules={{ required: "Please enter a unique agent name" }}
      >
        {({ value, onChange, ref, ...control }) => (
          <Input
            {...control}
            ref={ref}
            placeholder="e.g., my-langgraph-agent"
            value={typeof value === "string" ? value : ""}
            onChange={onChange}
          />
        )}
      </AgentFormField>

      <AgentFormField
        name="description"
        label={labelWithHint("Description", "Brief description of what this agent does")}
      >
        {({ value, onChange, ref, ...control }) => (
          <Textarea
            {...control}
            ref={ref}
            rows={2}
            placeholder="Describe what this agent does..."
            value={typeof value === "string" ? value : ""}
            onChange={onChange}
          />
        )}
      </AgentFormField>

      {agentTypeInfo.credential_fields.map((field) => (
        <CredentialField key={field.key} field={field} />
      ))}
    </FieldGroup>

    <div className="mb-4 rounded-md border border-border px-4">
      <AgentFormPanel panelKey={AGENT_FORM_CONFIG.cost.key} title={AGENT_FORM_CONFIG.cost.title} panels={panels}>
        <CostConfigFields />
      </AgentFormPanel>
    </div>
  </>
);

export const buildDynamicAgentData = (values: AgentFormValues, agentTypeInfo: AgentCreateInfo): AgentRequestPayload => {
  const litellmParams: Record<string, unknown> = {
    ...(agentTypeInfo.litellm_params_template || {}),
  };

  for (const field of agentTypeInfo.credential_fields) {
    const value = values[field.key];
    if (value && field.include_in_litellm_params !== false) {
      litellmParams[field.key] = value;
    }
  }

  if (values.cost_per_query) {
    litellmParams.cost_per_query = parseFloat(String(values.cost_per_query));
  }
  if (values.input_cost_per_token) {
    litellmParams.input_cost_per_token = parseFloat(String(values.input_cost_per_token));
  }
  if (values.output_cost_per_token) {
    litellmParams.output_cost_per_token = parseFloat(String(values.output_cost_per_token));
  }

  if (agentTypeInfo.model_template) {
    litellmParams.model = agentTypeInfo.credential_fields.reduce((model, field) => {
      const placeholder = `{${field.key}}`;
      const value = values[field.key];
      return model.includes(placeholder) && value ? model.replace(placeholder, String(value)) : model;
    }, agentTypeInfo.model_template);
  }

  const agentData: AgentRequestPayload = {
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
      skills: [
        {
          id: "chat",
          name: "Chat",
          description: "General chat capability",
          tags: ["chat", "conversation"],
        },
      ],
    },
    litellm_params: litellmParams,
  };

  if (values.tpm_limit != null) agentData.tpm_limit = values.tpm_limit;
  if (values.rpm_limit != null) agentData.rpm_limit = values.rpm_limit;
  if (values.session_tpm_limit != null) agentData.session_tpm_limit = values.session_tpm_limit;
  if (values.session_rpm_limit != null) agentData.session_rpm_limit = values.session_rpm_limit;

  return agentData;
};

export default DynamicAgentFormFields;
