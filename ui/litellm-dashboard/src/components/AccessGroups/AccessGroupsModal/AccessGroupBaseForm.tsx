import { useAgents } from "@/app/(dashboard)/hooks/agents/useAgents";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { ModelSelect } from "@/components/ModelSelect/ModelSelect";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";
import { BotIcon, InfoIcon, LayersIcon, ServerIcon } from "lucide-react";
import { useMemo } from "react";
import { Controller, useFormContext } from "react-hook-form";

export interface AccessGroupFormValues {
  name: string;
  description: string;
  modelIds: string[];
  mcpServerIds: string[];
  agentIds: string[];
}

interface AccessGroupBaseFormProps {
  isNameDisabled?: boolean;
}

/**
 * Multi-select rendered with shadcn Select + chip list below. Accepts any
 * `Array<{ label: string; value: string }>` list of options.
 */
function MultiSelect({
  value,
  onChange,
  options,
  placeholder,
  emptyText,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
  emptyText: string;
}) {
  const selected = useMemo(() => value ?? [], [value]);
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o.value)),
    [options, selected],
  );

  return (
    <div className="space-y-2">
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              {emptyText}
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge key={v} variant="secondary" className="flex items-center gap-1">
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() => onChange(selected.filter((s) => s !== v))}
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

/**
 * Fields-only block — expected to be rendered inside a parent form managed
 * by `react-hook-form` via a `FormProvider`. See `AccessGroupCreateModal`
 * / `AccessGroupEditModal` for the integration.
 */
export function AccessGroupBaseForm({
  isNameDisabled = false,
}: AccessGroupBaseFormProps) {
  const { control, register, formState } = useFormContext<AccessGroupFormValues>();
  const { data: agentsData } = useAgents();
  const { data: mcpServersData } = useMCPServers();

  const agents = agentsData?.agents ?? [];
  const mcpServers = mcpServersData ?? [];

  const mcpOptions = mcpServers.map((s) => ({
    label: s.server_name ?? s.server_id,
    value: s.server_id,
  }));
  const agentOptions = agents.map((a) => ({
    label: a.agent_name,
    value: a.agent_id,
  }));

  return (
    <Tabs defaultValue="general" className="w-full">
      <TabsList>
        <TabsTrigger value="general" className="gap-1">
          <InfoIcon size={16} />
          General Info
        </TabsTrigger>
        <TabsTrigger value="models" className="gap-1">
          <LayersIcon size={16} />
          Models
        </TabsTrigger>
        <TabsTrigger value="mcp" className="gap-1">
          <ServerIcon size={16} />
          MCP Servers
        </TabsTrigger>
        <TabsTrigger value="agents" className="gap-1">
          <BotIcon size={16} />
          Agents
        </TabsTrigger>
      </TabsList>

      <TabsContent value="general" className="pt-4 space-y-4">
        <div className="space-y-2">
          <Label htmlFor="access-group-name">
            Group Name <span className="text-destructive">*</span>
          </Label>
          <Input
            id="access-group-name"
            placeholder="e.g. Engineering Team"
            disabled={isNameDisabled}
            aria-invalid={!!formState.errors.name}
            {...register("name", {
              required: "Please enter the access group name",
            })}
          />
          {formState.errors.name && (
            <p className="text-sm text-destructive">
              {formState.errors.name.message as string}
            </p>
          )}
        </div>
        <div className="space-y-2">
          <Label htmlFor="access-group-description">Description</Label>
          <Textarea
            id="access-group-description"
            rows={4}
            placeholder="Describe the purpose of this access group..."
            {...register("description")}
          />
        </div>
      </TabsContent>

      <TabsContent value="models" className="pt-4">
        <div className="space-y-2">
          <Label>Allowed Models</Label>
          <Controller
            control={control}
            name="modelIds"
            render={({ field }) => (
              <ModelSelect
                context="global"
                value={field.value ?? []}
                onChange={(values) => field.onChange(values)}
                style={{ width: "100%" }}
              />
            )}
          />
        </div>
      </TabsContent>

      <TabsContent value="mcp" className="pt-4">
        <div className="space-y-2">
          <Label>Allowed MCP Servers</Label>
          <Controller
            control={control}
            name="mcpServerIds"
            render={({ field }) => (
              <MultiSelect
                value={field.value ?? []}
                onChange={field.onChange}
                options={mcpOptions}
                placeholder="Select MCP servers"
                emptyText="No MCP servers available"
              />
            )}
          />
        </div>
      </TabsContent>

      <TabsContent value="agents" className="pt-4">
        <div className="space-y-2">
          <Label>Allowed Agents</Label>
          <Controller
            control={control}
            name="agentIds"
            render={({ field }) => (
              <MultiSelect
                value={field.value ?? []}
                onChange={field.onChange}
                options={agentOptions}
                placeholder="Select agents"
                emptyText="No agents available"
              />
            )}
          />
        </div>
      </TabsContent>
    </Tabs>
  );
}
