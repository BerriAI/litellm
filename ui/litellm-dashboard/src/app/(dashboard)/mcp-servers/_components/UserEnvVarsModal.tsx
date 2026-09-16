import React from "react";
import { CircleAlert, Info } from "lucide-react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { z } from "zod/v4";
import { MCPServer, MCPUserEnvVarsStatus, MCPUserEnvVarSpec } from "@/components/mcp_tools/types";
import { getMCPUserEnvVars, storeMCPUserEnvVars } from "@/components/networking";
import { toast } from "@/lib/toast";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Badge } from "@/components/ui/badge";
import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";

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
                {spec.is_set && <Badge variant="secondary">Set</Badge>}
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
    <Dialog open={open} onOpenChange={(opened) => !opened && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[520px]">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <DialogTitle className="text-base font-semibold">Set your credentials</DialogTitle>
            <StatusBadge tone="info" label="Per-user" />
          </div>
          <span className="text-xs text-muted-foreground">{displayName}</span>
        </DialogHeader>

        <div className="mt-2 space-y-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <UiLoadingSpinner className="size-5" />
            </div>
          ) : isError ? (
            <Alert variant="error">
              <CircleAlert />
              <AlertTitle>Failed to load env vars</AlertTitle>
            </Alert>
          ) : required.length === 0 ? (
            <Alert variant="info">
              <Info />
              <AlertTitle>No per-user fields configured for this server.</AlertTitle>
            </Alert>
          ) : (
            <>
              <span className="block text-sm text-muted-foreground">
                These values are private to you. Your admin configured this MCP server to require these per-user
                credentials. Saved values are never shown back; leave an already-set field blank to keep it, or enter a
                value to set or change it.
              </span>
              <UserEnvVarsForm required={required} isSaving={isSaving} onCancel={onClose} onSubmit={handleSave} />
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default UserEnvVarsModal;
