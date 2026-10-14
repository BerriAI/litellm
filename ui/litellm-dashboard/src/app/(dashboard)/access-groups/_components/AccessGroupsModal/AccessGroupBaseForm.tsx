"use client";

import { BotIcon, InfoIcon, LayersIcon, ServerIcon } from "lucide-react";
import type { UseFormReturn } from "react-hook-form";
import { z } from "zod";

import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import MCPToolPermissions from "@/components/mcp_server_management/MCPToolPermissions";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";

export const accessGroupFormSchema = z.object({
  name: z.string().min(1, "Please enter the access group name"),
  description: z.string(),
  modelIds: z.array(z.string()),
  mcpServerIds: z.array(z.string()),
  mcpToolPermissions: z.record(z.string(), z.array(z.string())),
  agentIds: z.array(z.string()),
});

export type AccessGroupFormValues = z.output<typeof accessGroupFormSchema>;

export const GENERAL_TAB = "general";
export const MODELS_TAB = "models";
export const MCP_SERVERS_TAB = "mcp-servers";
export const AGENTS_TAB = "agents";

interface AccessGroupBaseFormProps {
  form: UseFormReturn<AccessGroupFormValues>;
  isNameDisabled?: boolean;
  activeTab: string;
  onTabChange: (tab: string) => void;
  initialServerIds?: readonly string[];
}

export function AccessGroupMCPToolPermissions({
  selectedServers,
  toolPermissions,
  onChange,
  preserveUnrestrictedServers,
}: {
  selectedServers: string[];
  toolPermissions: Record<string, string[]>;
  onChange: (next: Record<string, string[]>) => void;
  preserveUnrestrictedServers?: readonly string[];
}) {
  const { accessToken } = useAuthorized();

  if (selectedServers.length === 0) {
    return null;
  }

  return (
    <div className="pt-4">
      <MCPToolPermissions
        accessToken={accessToken ?? ""}
        selectedServers={selectedServers}
        toolPermissions={toolPermissions}
        onChange={onChange}
        preserveUnrestrictedServers={preserveUnrestrictedServers}
      />
    </div>
  );
}

export function AccessGroupBaseForm({
  form,
  isNameDisabled = false,
  activeTab,
  onTabChange,
  initialServerIds,
}: AccessGroupBaseFormProps) {
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
          <InfoIcon size={16} />
          General Info
        </TabsTrigger>
        <TabsTrigger value={MODELS_TAB}>
          <LayersIcon size={16} />
          Models
        </TabsTrigger>
        <TabsTrigger value={MCP_SERVERS_TAB}>
          <ServerIcon size={16} />
          MCP Servers
        </TabsTrigger>
        <TabsTrigger value={AGENTS_TAB}>
          <BotIcon size={16} />
          Agents
        </TabsTrigger>
      </TabsList>

      <TabsContent value={GENERAL_TAB} className="pt-4">
        <FieldGroup>
          <FormField control={form.control} name="name" label="Group Name">
            {({ ref, ...field }) => (
              <Input {...field} ref={ref} placeholder="e.g. Engineering Team" disabled={isNameDisabled} />
            )}
          </FormField>
          <FormField control={form.control} name="description" label="Description">
            {({ ref, ...field }) => (
              <Textarea {...field} ref={ref} rows={4} placeholder="Describe the purpose of this access group..." />
            )}
          </FormField>
        </FieldGroup>
      </TabsContent>

      <TabsContent value={MODELS_TAB} className="pt-4">
        <FormField control={form.control} name="modelIds" label="Allowed Models">
          {(field) => <ModelSelect context="global" value={field.value} onChange={field.onChange} />}
        </FormField>
      </TabsContent>

      <TabsContent value={MCP_SERVERS_TAB} className="pt-4">
        <FormField control={form.control} name="mcpServerIds" label="Allowed MCP Servers">
          {({ id, value, onChange }) => (
            <MultiSelect
              id={id}
              value={value}
              onValueChange={onChange}
              options={mcpServerOptions}
              placeholder="Select MCP servers"
            />
          )}
        </FormField>
        <AccessGroupMCPToolPermissions
          selectedServers={form.watch("mcpServerIds")}
          toolPermissions={form.watch("mcpToolPermissions")}
          onChange={(next) => form.setValue("mcpToolPermissions", next, { shouldDirty: true })}
          preserveUnrestrictedServers={initialServerIds}
        />
      </TabsContent>

      <TabsContent value={AGENTS_TAB} className="pt-4">
        <FormField control={form.control} name="agentIds" label="Allowed Agents">
          {({ id, value, onChange }) => (
            <MultiSelect
              id={id}
              value={value}
              onValueChange={onChange}
              options={agentOptions}
              placeholder="Select agents"
            />
          )}
        </FormField>
      </TabsContent>
    </Tabs>
  );
}
