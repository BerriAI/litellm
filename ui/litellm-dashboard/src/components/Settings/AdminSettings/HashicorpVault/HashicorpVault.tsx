"use client";

import { Edit, ExternalLink, Info, KeyRound, PlugZap, Trash2 } from "lucide-react";
import { useState } from "react";

import { testHashicorpVaultConnection } from "@/app/(dashboard)/hooks/configOverrides/hashicorpVaultApi";
import { useDeleteHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useDeleteHashicorpVaultConfig";
import { useHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig";
import { useUpdateHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

import EditHashicorpVaultModal from "./EditHashicorpVaultModal";
import HashicorpVaultEmptyPlaceholder from "./HashicorpVaultEmptyPlaceholder";
import { FIELD_LABELS, SENSITIVE_FIELDS } from "./constants";

function detectAuthMethod(values: Record<string, unknown>): string {
  if (values.approle_role_id || values.approle_secret_id) return "AppRole";
  if (values.client_cert && values.client_key) return "TLS Certificate";
  if (values.vault_token) return "Token";
  return "None";
}

function DetailRow({ children, label }: { children: React.ReactNode; label: string }) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-3">
      <dt className="bg-muted/50 px-4 py-3 text-sm font-medium text-foreground">{label}</dt>
      <dd className="px-4 py-3 text-sm text-foreground sm:col-span-2">{children}</dd>
    </div>
  );
}

export default function HashicorpVault() {
  const { accessToken } = useAuthorized();
  const { data, isLoading, isError, error } = useHashicorpVaultConfig();
  const { mutate: deleteConfig, isPending: isDeleting } = useDeleteHashicorpVaultConfig(accessToken);
  const { mutate: updateConfig, isPending: isClearingField } = useUpdateHashicorpVaultConfig(accessToken);
  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [clearingField, setClearingField] = useState<string | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const rawValues = data?.values ?? {};
  const isConfigured = Boolean(rawValues.vault_addr);

  const handleTestConnection = async () => {
    if (!accessToken) return;
    setIsTesting(true);
    try {
      const result = await testHashicorpVaultConnection(accessToken);
      toast.success(result.message || "Connection to Vault successful!");
    } catch (err) {
      toast.fromError(err);
    } finally {
      setIsTesting(false);
    }
  };

  const handleDelete = () => {
    deleteConfig(undefined, {
      onSuccess: () => {
        toast.success("Hashicorp Vault configuration deleted");
        setIsDeleteModalOpen(false);
      },
      onError: (err) => toast.fromError(err),
    });
  };

  const handleClearField = () => {
    if (!clearingField) return;
    updateConfig(
      { [clearingField]: "" },
      {
        onSuccess: () => {
          toast.success(`${FIELD_LABELS[clearingField] ?? clearingField} cleared`);
          setClearingField(null);
        },
        onError: (err) => toast.fromError(err),
      },
    );
  };

  const renderValue = (key: string) => {
    const value = rawValues[key];
    if (!value) return <span className="text-muted-foreground italic">Not configured</span>;
    if (!SENSITIVE_FIELDS.has(key)) return <span className="font-mono text-muted-foreground">{value}</span>;

    return (
      <div className="flex items-center justify-between gap-2">
        <span className="font-mono text-muted-foreground">{value}</span>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label={`Clear ${FIELD_LABELS[key] ?? key}`}
          onClick={() => setClearingField(key)}
        >
          <Trash2 className="size-3.5" />
        </Button>
      </div>
    );
  };

  const fieldsToShow = Object.entries(rawValues).filter(([, value]) => value != null && value !== "");

  return (
    <>
      {isLoading ? (
        <Card role="status" aria-label="Loading Hashicorp Vault configuration">
          <CardContent className="space-y-3">
            <Skeleton className="h-8 w-64" />
            <Skeleton className="h-40 w-full" />
          </CardContent>
        </Card>
      ) : isError ? (
        <Card>
          <CardContent>
            <Alert variant="error">
              <AlertTitle>Could not load Hashicorp Vault configuration</AlertTitle>
              {error instanceof Error && <AlertDescription>{error.message}</AlertDescription>}
            </Alert>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-3">
              <KeyRound className="size-6 text-muted-foreground" />
              <div>
                <CardTitle>
                  <h3>Hashicorp Vault</h3>
                </CardTitle>
                <CardDescription>Manage secret manager configuration</CardDescription>
              </div>
            </div>
            {isConfigured && (
              <CardAction className="flex flex-wrap gap-2">
                <Button type="button" variant="outline" disabled={isTesting} onClick={handleTestConnection}>
                  <PlugZap />
                  {isTesting ? "Testing..." : "Test Connection"}
                </Button>
                <Button type="button" variant="outline" onClick={() => setIsEditModalVisible(true)}>
                  <Edit />
                  Edit Configuration
                </Button>
                <Button type="button" variant="destructive" onClick={() => setIsDeleteModalOpen(true)}>
                  <Trash2 />
                  Delete Configuration
                </Button>
              </CardAction>
            )}
          </CardHeader>
          <CardContent className="space-y-6">
            {isConfigured && (
              <Alert variant="info">
                <Info />
                <AlertTitle>Secrets must be stored with the field name &quot;key&quot;</AlertTitle>
                <AlertDescription>
                  <code className="block font-mono">vault kv put secret/SECRET_NAME key=secret_value</code>
                  <a
                    href="https://docs.litellm.ai/docs/secret_managers/hashicorp_vault"
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1"
                  >
                    View documentation
                    <ExternalLink className="size-3" />
                  </a>
                </AlertDescription>
              </Alert>
            )}

            {isConfigured ? (
              fieldsToShow.length > 0 && (
                <dl className="divide-y divide-border overflow-hidden rounded-md border border-border">
                  <DetailRow label="Auth Method">{detectAuthMethod(rawValues)}</DetailRow>
                  {fieldsToShow.map(([key]) => (
                    <DetailRow key={key} label={FIELD_LABELS[key] ?? key}>
                      {renderValue(key)}
                    </DetailRow>
                  ))}
                </dl>
              )
            ) : (
              <HashicorpVaultEmptyPlaceholder onAdd={() => setIsEditModalVisible(true)} />
            )}
          </CardContent>
        </Card>
      )}

      <EditHashicorpVaultModal
        isVisible={isEditModalVisible}
        onCancel={() => setIsEditModalVisible(false)}
        onSuccess={() => setIsEditModalVisible(false)}
      />
      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title="Delete Hashicorp Vault Configuration?"
        message="Models using Vault secrets will lose access to their API keys until a new configuration is saved."
        resourceInformationTitle="Vault Configuration"
        resourceInformation={[{ label: "Vault Address", value: rawValues.vault_addr }]}
        onCancel={() => setIsDeleteModalOpen(false)}
        onOk={handleDelete}
        confirmLoading={isDeleting}
      />
      <DeleteResourceModal
        isOpen={clearingField !== null}
        title={`Clear ${clearingField ? FIELD_LABELS[clearingField] ?? clearingField : ""}?`}
        message="This will remove the stored value."
        resourceInformationTitle="Field"
        resourceInformation={[
          { label: "Field", value: clearingField ? FIELD_LABELS[clearingField] ?? clearingField : "" },
        ]}
        onCancel={() => setClearingField(null)}
        onOk={handleClearField}
        confirmLoading={isClearingField}
      />
    </>
  );
}
