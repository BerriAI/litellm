"use client";

import { BotIcon, InfoIcon, LayersIcon, ServerIcon } from "lucide-react";
import type { Control } from "react-hook-form";

import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

import type { AccessGroupFormValues } from "./schema";

export const GENERAL_TAB = "general";

interface MultiSelectOption {
  value: string;
  label: string;
}

interface MultiSelectProps {
  id: string;
  value: string[];
  onChange: (value: string[]) => void;
  options: MultiSelectOption[];
  placeholder: string;
  "aria-invalid": true | undefined;
  "aria-describedby": string | undefined;
}

const MultiSelect = ({
  id,
  value,
  onChange,
  options,
  placeholder,
  "aria-invalid": ariaInvalid,
  "aria-describedby": ariaDescribedBy,
}: MultiSelectProps) => (
  <Select multiple items={options} value={value} onValueChange={onChange}>
    <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy} className="w-full">
      <SelectValue placeholder={placeholder}>
        {(selected: string[]) =>
          selected.length === 0
            ? placeholder
            : options
                .filter((option) => selected.includes(option.value))
                .map((option) => option.label)
                .join(", ")
        }
      </SelectValue>
    </SelectTrigger>
    <SelectContent>
      {options.map((option) => (
        <SelectItem key={option.value} value={option.value}>
          {option.label}
        </SelectItem>
      ))}
    </SelectContent>
  </Select>
);

interface AccessGroupFormFieldsProps {
  control: Control<AccessGroupFormValues>;
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export const AccessGroupFormFields = ({ control, activeTab, onTabChange }: AccessGroupFormFieldsProps) => {
  const { data: agentsData } = useAgents();
  const { data: mcpServersData } = useMCPServers();

  const mcpServerOptions = (mcpServersData ?? []).map((server) => ({
    value: server.server_id,
    label: server.server_name ?? server.server_id,
  }));
  const agentOptions = (agentsData?.agents ?? []).map((agent) => ({
    value: agent.agent_id,
    label: agent.agent_name,
  }));

  return (
    <Tabs value={activeTab} onValueChange={onTabChange}>
      <TabsList className="w-full">
        <TabsTrigger value={GENERAL_TAB}>
          <InfoIcon />
          General Info
        </TabsTrigger>
        <TabsTrigger value="models">
          <LayersIcon />
          Models
        </TabsTrigger>
        <TabsTrigger value="mcp-servers">
          <ServerIcon />
          MCP Servers
        </TabsTrigger>
        <TabsTrigger value="agents">
          <BotIcon />
          Agents
        </TabsTrigger>
      </TabsList>

      <TabsContent value={GENERAL_TAB} className="pt-4">
        <FieldGroup>
          <FormField control={control} name="name" label="Group Name">
            {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="e.g. Engineering Team" />}
          </FormField>
          <FormField control={control} name="description" label="Description">
            {({ ref, ...field }) => (
              <Textarea {...field} ref={ref} rows={4} placeholder="Describe the purpose of this access group..." />
            )}
          </FormField>
        </FieldGroup>
      </TabsContent>

      <TabsContent value="models" className="pt-4">
        <FormField control={control} name="modelIds" label="Allowed Models">
          {(field) => <ModelSelect context="global" value={field.value} onChange={field.onChange} />}
        </FormField>
      </TabsContent>

      <TabsContent value="mcp-servers" className="pt-4">
        <FormField control={control} name="mcpServerIds" label="Allowed MCP Servers">
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <MultiSelect
              id={id}
              value={value}
              onChange={onChange}
              options={mcpServerOptions}
              placeholder="Select MCP servers"
              aria-invalid={ariaInvalid}
              aria-describedby={ariaDescribedBy}
            />
          )}
        </FormField>
      </TabsContent>

      <TabsContent value="agents" className="pt-4">
        <FormField control={control} name="agentIds" label="Allowed Agents">
          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
            <MultiSelect
              id={id}
              value={value}
              onChange={onChange}
              options={agentOptions}
              placeholder="Select agents"
              aria-invalid={ariaInvalid}
              aria-describedby={ariaDescribedBy}
            />
          )}
        </FormField>
      </TabsContent>
    </Tabs>
  );
};
