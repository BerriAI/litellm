"use client";

import { useState } from "react";
import { useHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig";
import { useDeleteHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useDeleteHashicorpVaultConfig";
import { useUpdateHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import NotificationManager from "@/components/molecules/notifications_manager";
import { testHashicorpVaultConnection } from "@/app/(dashboard)/hooks/configOverrides/hashicorpVaultApi";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Edit, KeyRound, PlugZap, Trash2 } from "lucide-react";
import { SENSITIVE_FIELDS, FIELD_LABELS } from "./constants";
import EditHashicorpVaultModal from "./EditHashicorpVaultModal";
import HashicorpVaultEmptyPlaceholder from "./HashicorpVaultEmptyPlaceholder";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function detectAuthMethod(values: Record<string, any>): string {
  if (values.approle_role_id || values.approle_secret_id) return "AppRole";
  if (values.client_cert && values.client_key) return "TLS Certificate";
  if (values.vault_token) return "Token";
  return "None";
}

export default function HashicorpVault() {
  const { accessToken } = useAuthorized();
  const { data, isLoading, isError, error } = useHashicorpVaultConfig();
  const { mutate: deleteConfig, isPending: isDeleting } =
    useDeleteHashicorpVaultConfig(accessToken);
  const { mutate: updateConfig, isPending: isClearingField } =
    useUpdateHashicorpVaultConfig(accessToken);

  const [isEditModalVisible, setIsEditModalVisible] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [clearingField, setClearingField] = useState<string | null>(null);
  const [isTesting, setIsTesting] = useState(false);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const rawValues: Record<string, any> = data?.values ?? {};
  const isConfigured = Boolean(rawValues.vault_addr);

  const handleTestConnection = async () => {
    if (!accessToken) return;
    setIsTesting(true);
    try {
      const result = await testHashicorpVaultConnection(accessToken);
      NotificationManager.success(
        result.message || "Connection to Vault successful!",
      );
    } catch (err) {
      NotificationManager.fromBackend(err);
    } finally {
      setIsTesting(false);
    }
  };

  const handleDelete = () => {
    deleteConfig(undefined, {
      onSuccess: () => {
        NotificationManager.success(
          "Hashicorp Vault configuration deleted",
        );
        setIsDeleteModalOpen(false);
      },
      onError: (err) => {
        NotificationManager.fromBackend(err);
      },
    });
  };

  const handleClearField = () => {
    if (!clearingField) return;
    updateConfig(
      { [clearingField]: "" },
      {
        onSuccess: () => {
          NotificationManager.success(
            `${FIELD_LABELS[clearingField] ?? clearingField} cleared`,
          );
          setClearingField(null);
        },
        onError: (err) => {
          NotificationManager.fromBackend(err);
        },
      },
    );
  };

  const renderValue = (key: string) => {
    const value = rawValues[key];
    if (!value) {
      return (
        <span className="text-muted-foreground italic">Not configured</span>
      );
    }
    if (SENSITIVE_FIELDS.has(key)) {
      return (
        <div className="flex justify-between items-center">
          <span className="font-mono text-muted-foreground">{value}</span>
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 text-destructive hover:text-destructive"
            onClick={() => setClearingField(key)}
            aria-label={`Clear ${FIELD_LABELS[key] ?? key}`}
          >
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </div>
      );
    }
    return <span className="font-mono text-muted-foreground">{value}</span>;
  };

  const renderSettings = () => {
    const fieldsToShow = Object.entries(rawValues).filter(
      ([, value]) => value != null && value !== "",
    );
    if (fieldsToShow.length === 0) return null;
    return (
      <dl className="border border-border rounded-md overflow-hidden text-sm">
        <div className="grid grid-cols-[200px_1fr] border-b border-border">
          <dt className="bg-muted px-4 py-3 font-medium">Auth Method</dt>
          <dd className="px-4 py-3">{detectAuthMethod(rawValues)}</dd>
        </div>
        {fieldsToShow.map(([key], i) => (
          <div
            key={key}
            className={`grid grid-cols-[200px_1fr] ${
              i < fieldsToShow.length - 1 ? "border-b border-border" : ""
            }`}
          >
            <dt className="bg-muted px-4 py-3 font-medium">
              {FIELD_LABELS[key] ?? key}
            </dt>
            <dd className="px-4 py-3">{renderValue(key)}</dd>
          </div>
        ))}
      </dl>
    );
  };

  return (
    <>
      {isLoading ? (
        <Card className="p-6">
          <Skeleton className="h-32 w-full" />
        </Card>
      ) : isError ? (
        <Card className="p-6">
          <Alert variant="destructive">
            <AlertTitle>Could not load Hashicorp Vault configuration</AlertTitle>
            {error instanceof Error && (
              <AlertDescription>{error.message}</AlertDescription>
            )}
          </Alert>
        </Card>
      ) : (
        <Card className="p-6">
          <div className="space-y-6 w-full">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-3">
                <KeyRound className="w-6 h-6 text-muted-foreground" />
                <div>
                  <h3 className="text-lg font-semibold m-0">
                    Hashicorp Vault
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    Manage secret manager configuration
                  </p>
                </div>
              </div>

              <div className="flex gap-2">
                {isConfigured && (
                  <>
                    <Button
                      variant="outline"
                      onClick={handleTestConnection}
                      disabled={isTesting}
                    >
                      <PlugZap className="w-4 h-4" />
                      {isTesting ? "Testing…" : "Test Connection"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => setIsEditModalVisible(true)}
                    >
                      <Edit className="w-4 h-4" />
                      Edit Configuration
                    </Button>
                    <Button
                      variant="destructive"
                      onClick={() => setIsDeleteModalOpen(true)}
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete Configuration
                    </Button>
                  </>
                )}
              </div>
            </div>

            {isConfigured && (
              <Alert>
                <AlertTitle>
                  Secrets must be stored with the field name &quot;key&quot;
                </AlertTitle>
                <AlertDescription>
                  <code className="bg-muted px-1 py-0.5 rounded text-xs">
                    vault kv put secret/SECRET_NAME key=secret_value
                  </code>
                  <br />
                  <a
                    href="https://docs.litellm.ai/docs/secret_managers/hashicorp_vault"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    View documentation
                  </a>
                </AlertDescription>
              </Alert>
            )}

            {isConfigured ? (
              renderSettings()
            ) : (
              <HashicorpVaultEmptyPlaceholder
                onAdd={() => setIsEditModalVisible(true)}
              />
            )}
          </div>
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
        resourceInformation={[
          { label: "Vault Address", value: rawValues.vault_addr },
        ]}
        onCancel={() => setIsDeleteModalOpen(false)}
        onOk={handleDelete}
        confirmLoading={isDeleting}
      />

      <DeleteResourceModal
        isOpen={clearingField !== null}
        title={`Clear ${clearingField ? (FIELD_LABELS[clearingField] ?? clearingField) : ""}?`}
        message="This will remove the stored value."
        resourceInformationTitle="Field"
        resourceInformation={[
          {
            label: "Field",
            value: clearingField
              ? (FIELD_LABELS[clearingField] ?? clearingField)
              : "",
          },
        ]}
        onCancel={() => setClearingField(null)}
        onOk={handleClearField}
        confirmLoading={isClearingField}
      />
    </>
  );
}
