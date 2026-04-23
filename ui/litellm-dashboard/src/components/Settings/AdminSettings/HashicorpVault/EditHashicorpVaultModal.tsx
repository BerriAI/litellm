"use client";

import { useHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig";
import { useUpdateHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import NotificationManager from "@/components/molecules/notifications_manager";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Eye, EyeOff } from "lucide-react";
import React, { useEffect, useState } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { SENSITIVE_FIELDS, FIELD_LABELS } from "./constants";

interface FieldGroup {
  title: string;
  subtitle?: string;
  fields: string[];
}

const FIELD_GROUPS: FieldGroup[] = [
  {
    title: "Connection",
    fields: [
      "vault_addr",
      "vault_namespace",
      "vault_mount_name",
      "vault_path_prefix",
    ],
  },
  {
    title: "Token Authentication",
    subtitle:
      "Use a Vault token to authenticate. Only one auth method is required.",
    fields: ["vault_token"],
  },
  {
    title: "AppRole Authentication",
    subtitle:
      "Use AppRole credentials to authenticate. Only one auth method is required.",
    fields: ["approle_role_id", "approle_secret_id", "approle_mount_path"],
  },
  {
    title: "TLS",
    subtitle: "Optional client certificate for mTLS.",
    fields: ["client_cert", "client_key", "vault_cert_role"],
  },
];

interface EditHashicorpVaultModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type FormValues = Record<string, any>;

const EditHashicorpVaultModal: React.FC<EditHashicorpVaultModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
}) => {
  const { accessToken } = useAuthorized();
  const { data } = useHashicorpVaultConfig();
  const { mutate, isPending } = useUpdateHashicorpVaultConfig(accessToken);
  const form = useForm<FormValues>({ mode: "onSubmit" });
  const [showSensitive, setShowSensitive] = useState<Record<string, boolean>>(
    {},
  );

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const schema: any = data?.field_schema;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const properties: Record<string, any> = schema?.properties ?? {};
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const rawValues: Record<string, any> = data?.values ?? {};

  useEffect(() => {
    if (isVisible && data) {
      const formValues: FormValues = {};
      for (const [key, value] of Object.entries(rawValues)) {
        if (!SENSITIVE_FIELDS.has(key)) formValues[key] = value;
      }
      form.reset(formValues);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVisible, data]);

  const handleSubmit = form.handleSubmit((formValues) => {
    const config: FormValues = {};
    for (const [key, value] of Object.entries(formValues)) {
      if (value !== undefined && value !== null && value !== "") {
        config[key] = value;
      } else if (!SENSITIVE_FIELDS.has(key)) {
        config[key] = "";
      }
    }
    mutate(config, {
      onSuccess: () => {
        NotificationManager.success(
          "Hashicorp Vault configuration updated successfully",
        );
        onSuccess();
      },
      onError: (err) => {
        NotificationManager.fromBackend(err);
      },
    });
  });

  const handleCancel = () => {
    form.reset();
    onCancel();
  };

  const renderField = (fieldName: string) => {
    const fieldSchema = properties[fieldName];
    if (!fieldSchema) return null;

    const isSensitive = SENSITIVE_FIELDS.has(fieldName);
    const existingValue = rawValues[fieldName];
    const hasExistingValue =
      isSensitive && existingValue != null && existingValue !== "";
    const placeholder = hasExistingValue
      ? `Leave blank to keep existing (${existingValue})`
      : fieldSchema?.description;

    const validators =
      fieldName === "vault_addr"
        ? {
            pattern: {
              value: /^https?:\/\/.+/,
              message: "Must start with http:// or https://",
            },
          }
        : {};

    return (
      <div key={fieldName} className="space-y-2 mb-4">
        <Label htmlFor={fieldName}>
          {FIELD_LABELS[fieldName] ?? fieldName}
        </Label>
        {isSensitive ? (
          <div className="relative">
            <Input
              id={fieldName}
              type={showSensitive[fieldName] ? "text" : "password"}
              placeholder={placeholder}
              {...form.register(fieldName, validators)}
            />
            <button
              type="button"
              onClick={() =>
                setShowSensitive((prev) => ({
                  ...prev,
                  [fieldName]: !prev[fieldName],
                }))
              }
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground"
              aria-label={
                showSensitive[fieldName] ? "Hide value" : "Show value"
              }
            >
              {showSensitive[fieldName] ? (
                <EyeOff size={14} />
              ) : (
                <Eye size={14} />
              )}
            </button>
          </div>
        ) : (
          <Input
            id={fieldName}
            placeholder={fieldSchema?.description}
            {...form.register(fieldName, validators)}
          />
        )}
        {form.formState.errors[fieldName] && (
          <p className="text-sm text-destructive">
            {form.formState.errors[fieldName]?.message as string}
          </p>
        )}
      </div>
    );
  };

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Edit Hashicorp Vault Configuration</DialogTitle>
          <DialogDescription className="sr-only">
            Configure Hashicorp Vault credentials and connection details.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={handleSubmit}>
            {FIELD_GROUPS.map((group, index) => (
              <div key={group.title}>
                {index > 0 && <Separator className="my-4" />}
                <h5 className="text-base font-semibold mb-1">
                  {group.title}
                </h5>
                {group.subtitle && (
                  <p className="text-sm text-muted-foreground mb-4">
                    {group.subtitle}
                  </p>
                )}
                {group.fields.map(renderField)}
              </div>
            ))}
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={handleCancel}
                disabled={isPending}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isPending}>
                {isPending ? "Saving..." : "Save"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default EditHashicorpVaultModal;
