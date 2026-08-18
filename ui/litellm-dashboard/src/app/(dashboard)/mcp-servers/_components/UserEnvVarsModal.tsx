import React from "react";
import { Modal, Alert, Spin, Tag, Typography } from "antd";
import { useMutation, useQuery } from "@tanstack/react-query";
import { z } from "zod/v4";
import { MCPServer, MCPUserEnvVarsStatus, MCPUserEnvVarSpec } from "@/components/mcp_tools/types";
import { getMCPUserEnvVars, storeMCPUserEnvVars } from "@/components/networking";
import { toast } from "@/lib/toast";
import { FieldGroup } from "@/components/shared/form/field";
import { FormField } from "@/components/shared/form/FormField";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";

const { Text, Title } = Typography;

interface UserEnvVarsModalProps {
  server: MCPServer | null;
  open: boolean;
  accessToken: string | null;
  onClose: () => void;
  onSaved?: (status: MCPUserEnvVarsStatus) => void;
}

interface UserEnvVarsFormProps {
  required: readonly MCPUserEnvVarSpec[];
  isSaving: boolean;
  onCancel: () => void;
  onSubmit: (values: Record<string, string>) => void;
}

const buildSchema = (required: readonly MCPUserEnvVarSpec[]) =>
  z.object(
    Object.fromEntries(
      required.map((spec) => [spec.name, spec.is_set ? z.string() : z.string().min(1, `${spec.name} is required`)]),
    ),
  );

const emptyValues = (required: readonly MCPUserEnvVarSpec[]): Record<string, string> =>
  Object.fromEntries(required.map((spec) => [spec.name, ""]));

const UserEnvVarsForm: React.FC<UserEnvVarsFormProps> = ({ required, isSaving, onCancel, onSubmit }) => {
  const form = useZodForm(buildSchema(required), { defaultValues: emptyValues(required) });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <FieldGroup>
        {required.map((spec) => (
          <FormField
            key={spec.name}
            control={form.control}
            name={spec.name}
            description={spec.description || undefined}
            label={
              <span className="flex items-center gap-2">
                <span className="font-mono text-sm font-semibold">{spec.name}</span>
                {spec.is_set && <Tag color="green">Set</Tag>}
              </span>
            }
          >
            {(field) => (
              <PasswordInput
                {...field}
                disabled={isSaving}
                placeholder={
                  spec.is_set ? "Enter a new value to overwrite" : spec.description || `Enter your ${spec.name}`
                }
              />
            )}
          </FormField>
        ))}
      </FieldGroup>
      <div className="mt-6 flex items-center justify-end gap-2 border-t border-border pt-2">
        <Button type="button" variant="outline" onClick={onCancel} disabled={isSaving}>
          Cancel
        </Button>
        <Button type="submit" disabled={isSaving}>
          {isSaving && <UiLoadingSpinner className="mr-2 size-4" />}
          Save Credentials
        </Button>
      </div>
    </form>
  );
};

/**
 * User-facing modal for filling in per-user MCP environment variables.
 *
 * Backed by GET / POST ``/v1/mcp/server/{id}/user-env-vars``. Each field
 * the admin marked as ``scope=user`` shows up with the admin-supplied
 * description as the placeholder.
 */
const UserEnvVarsModal: React.FC<UserEnvVarsModalProps> = ({ server, open, accessToken, onClose, onSaved }) => {
  const [formGeneration, setFormGeneration] = React.useState(0);

  const {
    data: status,
    isLoading,
    isError,
  } = useQuery<MCPUserEnvVarsStatus>({
    queryKey: ["mcpUserEnvVars", server?.server_id],
    queryFn: () => getMCPUserEnvVars(accessToken!, server!.server_id),
    enabled: open && !!server && !!accessToken,
  });

  const saveMutation = useMutation({
    mutationFn: (values: Record<string, string>) => storeMCPUserEnvVars(accessToken!, server!.server_id, values),
    onSuccess: (saved) => {
      toast.success("Credentials saved");
      onSaved?.(saved);
      onClose();
    },
    onError: (err) => {
      toast.fromError(`Failed to save env vars: ${err instanceof Error ? err.message : String(err)}`);
    },
  });

  const handleSave = (values: Record<string, string>) => {
    if (!server || !accessToken) return;
    const trimmed: Record<string, string> = {};
    for (const [k, v] of Object.entries(values)) {
      trimmed[k] = (v ?? "").trim();
    }
    saveMutation.mutate(trimmed);
  };

  const displayName = server?.server_name || server?.alias || server?.server_id || "MCP Server";
  const required = status?.required ?? [];
  const isSaving = saveMutation.isPending;

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={520}
      destroyOnHidden
      afterOpenChange={(opened) => {
        if (opened) setFormGeneration((generation) => generation + 1);
      }}
      title={
        <div>
          <div className="flex items-center gap-2">
            <Title level={5} style={{ margin: 0 }}>
              Set your credentials
            </Title>
            <Tag color="blue">Per-user</Tag>
          </div>
          <Text type="secondary" className="text-xs">
            {displayName}
          </Text>
        </div>
      }
    >
      <div className="space-y-4 mt-2">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Spin />
          </div>
        ) : isError ? (
          <Alert type="error" showIcon message="Failed to load env vars" />
        ) : required.length === 0 ? (
          <Alert type="info" showIcon message="No per-user fields configured for this server." />
        ) : (
          <>
            <Text className="text-sm text-muted-foreground block">
              These values are private to you. Your admin configured this MCP server to require these per-user
              credentials. Saved values are never shown back; leave an already-set field blank to keep it, or enter a
              value to set or change it.
            </Text>
            <UserEnvVarsForm
              key={formGeneration}
              required={required}
              isSaving={isSaving}
              onCancel={onClose}
              onSubmit={handleSave}
            />
          </>
        )}
      </div>
    </Modal>
  );
};

export default UserEnvVarsModal;
