import { useId } from "react";
import { Controller, useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import type { MCPServer } from "@/components/mcp_tools/types";
import { createMCPServer, updateMCPServer } from "@/components/networking";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription, FieldError, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "@/lib/toast";
import { DUPLICATE_IDENTIFIER_MESSAGE, findDuplicateMcpServer, mcpSubmitErrorReason } from "./duplicateServerCheck";
import {
  buildProviderRegistrationPayload,
  providerRegistrationDefaults,
  providerRegistrationSchema,
  type ProviderRegistrationValues,
} from "./providerRegistrationForm";

interface ProviderMCPRegistrationFormProps {
  readonly mcpServer?: MCPServer;
  readonly accessToken: string | null;
  readonly availableAccessGroups: string[];
  readonly existingServers?: MCPServer[];
  readonly onCancel: () => void;
  readonly onSuccess: (server: MCPServer) => void;
}

export default function ProviderMCPRegistrationForm({
  mcpServer,
  accessToken,
  availableAccessGroups,
  existingServers,
  onCancel,
  onSuccess,
}: ProviderMCPRegistrationFormProps) {
  const id = useId();
  const form = useForm<ProviderRegistrationValues>({
    resolver: zodResolver(providerRegistrationSchema),
    defaultValues: providerRegistrationDefaults(mcpServer),
  });
  const { errors, isSubmitting } = form.formState;
  const submitLabel = mcpServer ? "Save Changes" : "Add MCP Server";
  const save = async (values: ProviderRegistrationValues) => {
    if (!accessToken || mcpServer?.is_config) return;
    const duplicate = findDuplicateMcpServer(existingServers, values.server_name, values.alias, mcpServer?.server_id);
    if (duplicate) {
      form.setError(duplicate.field, { message: DUPLICATE_IDENTIFIER_MESSAGE });
      return;
    }
    try {
      const payload = buildProviderRegistrationPayload(values, mcpServer);
      const server = mcpServer
        ? await updateMCPServer(accessToken, payload)
        : await createMCPServer(accessToken, payload);
      toast.success(mcpServer ? "MCP server updated" : "MCP server registered");
      onSuccess(server);
    } catch (error) {
      form.setError("root", { message: mcpSubmitErrorReason(error) });
    }
  };

  return (
    <form onSubmit={form.handleSubmit(save)} className="space-y-5">
      <p className="text-sm text-muted-foreground">
        The LLM provider discovers and calls this MCP server. LiteLLM checks server and tool permissions without
        connecting to it. Supply authentication in your provider request.
      </p>
      <div className="grid gap-5 sm:grid-cols-2">
        <Field data-invalid={!!errors.server_name}>
          <FieldLabel htmlFor={`${id}-name`}>MCP Server Name</FieldLabel>
          <Input
            id={`${id}-name`}
            {...form.register("server_name")}
            aria-invalid={!!errors.server_name}
            placeholder="docs_native"
          />
          <FieldError errors={[errors.server_name]} />
        </Field>
        <Field data-invalid={!!errors.alias}>
          <FieldLabel htmlFor={`${id}-alias`}>Alias (optional)</FieldLabel>
          <Input id={`${id}-alias`} {...form.register("alias")} aria-invalid={!!errors.alias} placeholder="docs" />
          <FieldError errors={[errors.alias]} />
        </Field>
      </div>
      <Field data-invalid={!!errors.url}>
        <FieldLabel htmlFor={`${id}-url`}>MCP Server URL</FieldLabel>
        <Input
          id={`${id}-url`}
          {...form.register("url")}
          aria-invalid={!!errors.url}
          placeholder="https://docs.example.com/mcp"
        />
        <FieldDescription>Use the same HTTPS endpoint in your provider request.</FieldDescription>
        <FieldError errors={[errors.url]} />
      </Field>
      <Field>
        <FieldLabel htmlFor={`${id}-description`}>Description (optional)</FieldLabel>
        <Input id={`${id}-description`} {...form.register("description")} placeholder="Search and read documentation" />
      </Field>
      <Field data-invalid={!!errors.tools_text}>
        <FieldLabel htmlFor={`${id}-tools`}>Allowed tools</FieldLabel>
        <Textarea
          id={`${id}-tools`}
          {...form.register("tools_text")}
          aria-invalid={!!errors.tools_text}
          aria-describedby={`${id}-tools-help`}
          placeholder={"search_docs\nfetch_doc"}
          rows={4}
          className="font-mono"
        />
        <FieldDescription id={`${id}-tools-help`}>
          Enter one exact tool name per line. An empty list blocks all tool calls. LiteLLM does not discover this list.
        </FieldDescription>
        <FieldError errors={[errors.tools_text]} />
      </Field>
      <Field>
        <FieldLabel htmlFor={`${id}-groups`}>Access groups</FieldLabel>
        <Controller
          control={form.control}
          name="mcp_access_groups"
          render={({ field }) => (
            <MultiSelect
              id={`${id}-groups`}
              options={availableAccessGroups.map((group) => ({ label: group, value: group }))}
              value={field.value}
              onValueChange={field.onChange}
              allowCustomValues
              placeholder="Select or create access groups"
            />
          )}
        />
      </Field>
      <Field orientation="horizontal">
        <Controller
          control={form.control}
          name="allow_all_keys"
          render={({ field }) => (
            <Switch id={`${id}-all-keys`} checked={field.value} onCheckedChange={field.onChange} />
          )}
        />
        <div>
          <FieldLabel htmlFor={`${id}-all-keys`}>Allow All LiteLLM Keys</FieldLabel>
          <FieldDescription>Otherwise, grant access through key, team, or toolset permissions.</FieldDescription>
        </div>
      </Field>
      <FieldError errors={[errors.root]} />
      <div className="flex justify-end gap-3 border-t pt-4">
        <Button type="button" variant="secondary" onClick={onCancel} disabled={isSubmitting}>
          Cancel
        </Button>
        <Button type="submit" disabled={isSubmitting || !accessToken || mcpServer?.is_config} aria-busy={isSubmitting}>
          {isSubmitting ? "Saving..." : submitLabel}
        </Button>
      </div>
    </form>
  );
}
