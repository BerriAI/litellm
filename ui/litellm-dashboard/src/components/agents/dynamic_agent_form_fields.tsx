import React, { useEffect } from "react";
import { Controller, useFormContext } from "react-hook-form";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import { AgentCreateInfo, AgentCredentialFieldMetadata } from "../networking";
import { AGENT_FORM_CONFIG } from "./agent_config";
import CostConfigFields from "./cost_config_fields";

interface DynamicAgentFormFieldsProps {
  agentTypeInfo: AgentCreateInfo;
}

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="ml-1 inline h-3 w-3 text-muted-foreground" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

/**
 * Form fields for dynamic agent types (e.g., LangGraph).
 * Renders common fields (agent name, display name, description) plus
 * credential fields defined by the agent type metadata.
 */
const DynamicAgentFormFields: React.FC<DynamicAgentFormFieldsProps> = ({
  agentTypeInfo,
}) => {
  const { register, control, setValue, formState, getValues } =
    useFormContext();

  useEffect(() => {
    for (const field of agentTypeInfo.credential_fields) {
      if (field.default_value !== undefined) {
        const current = getValues(field.key);
        if (current === undefined || current === null || current === "") {
          setValue(field.key, field.default_value);
        }
      }
    }
  }, [agentTypeInfo, setValue, getValues]);

  const agentNameError = (formState.errors as any)?.agent_name;

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="agent_name">
          Agent Name <span className="text-destructive">*</span>
          <InfoTip>Unique identifier for the agent</InfoTip>
        </Label>
        <Input
          id="agent_name"
          placeholder="e.g., my-langgraph-agent"
          aria-invalid={!!agentNameError}
          {...register("agent_name", {
            required: "Please enter a unique agent name",
          })}
        />
        {agentNameError && (
          <p className="text-sm text-destructive">
            {agentNameError.message as string}
          </p>
        )}
      </div>

      <div className="space-y-2">
        <Label htmlFor="description">
          Description
          <InfoTip>Brief description of what this agent does</InfoTip>
        </Label>
        <Textarea
          id="description"
          rows={2}
          placeholder="Describe what this agent does..."
          {...register("description")}
        />
      </div>

      {agentTypeInfo.credential_fields.map(
        (field: AgentCredentialFieldMetadata) => {
          const error = (formState.errors as any)?.[field.key];
          const inputId = `credential-${field.key}`;
          return (
            <div key={field.key} className="space-y-2">
              <Label htmlFor={inputId}>
                {field.label}
                {field.required && (
                  <span className="text-destructive"> *</span>
                )}
                {field.tooltip ? <InfoTip>{field.tooltip}</InfoTip> : null}
              </Label>
              {field.field_type === "password" ? (
                <Input
                  id={inputId}
                  type="password"
                  placeholder={field.placeholder || ""}
                  aria-invalid={!!error}
                  {...register(
                    field.key,
                    field.required
                      ? { required: `Please enter ${field.label}` }
                      : undefined,
                  )}
                />
              ) : field.field_type === "textarea" ? (
                <Textarea
                  id={inputId}
                  rows={3}
                  placeholder={field.placeholder || ""}
                  aria-invalid={!!error}
                  {...register(
                    field.key,
                    field.required
                      ? { required: `Please enter ${field.label}` }
                      : undefined,
                  )}
                />
              ) : field.field_type === "select" && field.options ? (
                <Controller
                  control={control}
                  name={field.key}
                  rules={
                    field.required
                      ? { required: `Please enter ${field.label}` }
                      : undefined
                  }
                  render={({ field: rhfField }) => (
                    <Select
                      value={(rhfField.value as string) ?? ""}
                      onValueChange={rhfField.onChange}
                    >
                      <SelectTrigger id={inputId} aria-invalid={!!error}>
                        <SelectValue
                          placeholder={field.placeholder || ""}
                        />
                      </SelectTrigger>
                      <SelectContent>
                        {field.options?.map((opt) => (
                          <SelectItem key={opt} value={opt}>
                            {opt}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              ) : (
                <Input
                  id={inputId}
                  placeholder={field.placeholder || ""}
                  aria-invalid={!!error}
                  {...register(
                    field.key,
                    field.required
                      ? { required: `Please enter ${field.label}` }
                      : undefined,
                  )}
                />
              )}
              {error && (
                <p className="text-sm text-destructive">
                  {error.message as string}
                </p>
              )}
            </div>
          );
        },
      )}

      <Accordion type="single" collapsible className="mb-4">
        <AccordionItem
          value={AGENT_FORM_CONFIG.cost.key}
          className="border border-border rounded-md px-3"
        >
          <AccordionTrigger className="py-2 hover:no-underline">
            {AGENT_FORM_CONFIG.cost.title}
          </AccordionTrigger>
          <AccordionContent>
            <CostConfigFields />
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
};

/**
 * Builds agent data from form values for dynamic agent types.
 * Uses configuration from agentTypeInfo to determine which fields to include.
 */
export const buildDynamicAgentData = (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  values: any,
  agentTypeInfo: AgentCreateInfo,
) => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const agentData: Record<string, any> = {
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

  if (values.tpm_limit != null) agentData.tpm_limit = values.tpm_limit;
  if (values.rpm_limit != null) agentData.rpm_limit = values.rpm_limit;
  if (values.session_tpm_limit != null) agentData.session_tpm_limit = values.session_tpm_limit;
  if (values.session_rpm_limit != null) agentData.session_rpm_limit = values.session_rpm_limit;

  return agentData;
};

export default DynamicAgentFormFields;
