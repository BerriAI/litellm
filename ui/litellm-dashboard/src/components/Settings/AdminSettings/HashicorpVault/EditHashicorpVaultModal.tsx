"use client";

import { useHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig";
import { useUpdateHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { toast } from "@/lib/toast";
import type { ParseKeys } from "i18next";
import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import { z } from "zod";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { Separator } from "@/components/ui/separator";
import { useZodForm } from "@/lib/forms/useZodForm";
import i18n from "@/i18n/i18n";
import { SENSITIVE_FIELDS, FIELD_LABELS } from "./constants";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface VaultFieldGroup {
  title: ParseKeys;
  subtitle?: ParseKeys;
  fields: string[];
}

const FIELD_GROUPS: VaultFieldGroup[] = [
  {
    title: "adminPanel.hashicorpVault.editModal.groups.connection.title",
    fields: [
      "vault_addr",
      "vault_namespace",
      "vault_login_namespace",
      "vault_secret_namespace",
      "vault_mount_name",
      "vault_path_prefix",
    ],
  },
  {
    title: "adminPanel.hashicorpVault.editModal.groups.token.title",
    subtitle: "adminPanel.hashicorpVault.editModal.groups.token.subtitle",
    fields: ["vault_token"],
  },
  {
    title: "adminPanel.hashicorpVault.editModal.groups.appRole.title",
    subtitle: "adminPanel.hashicorpVault.editModal.groups.appRole.subtitle",
    fields: ["approle_role_id", "approle_secret_id", "approle_mount_path"],
  },
  {
    title: "adminPanel.hashicorpVault.editModal.groups.tls.title",
    subtitle: "adminPanel.hashicorpVault.editModal.groups.tls.subtitle",
    fields: ["client_cert", "client_key", "vault_cert_role"],
  },
];

type VaultFormValues = Record<string, string>;

const buildSchema = (fields: readonly string[]): z.ZodType<VaultFormValues, VaultFormValues> =>
  z.object(
    Object.fromEntries(
      fields.map((name) => [
        name,
        name === "vault_addr"
          ? z.string().refine((value) => value.length === 0 || /^https?:\/\/.+/.test(value), {
              error: () => i18n.t("adminPanel.hashicorpVault.editModal.errors.invalidUrl"),
            })
          : z.string(),
      ]),
    ),
  ) as unknown as z.ZodType<VaultFormValues, VaultFormValues>;

interface EditHashicorpVaultModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const EditHashicorpVaultModal: React.FC<EditHashicorpVaultModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const { data } = useHashicorpVaultConfig();
  const { mutate, isPending } = useUpdateHashicorpVaultConfig(accessToken);

  const properties: Record<string, { description?: string }> = useMemo(
    () => data?.field_schema?.properties ?? {},
    [data],
  );
  const rawValues: Record<string, unknown> = useMemo(() => data?.values ?? {}, [data]);

  const visibleFields = useMemo(
    () => FIELD_GROUPS.flatMap((group) => group.fields).filter((name) => properties[name] !== undefined),
    [properties],
  );

  const seededValues = useMemo(
    () =>
      Object.fromEntries(
        visibleFields.map((name) => [name, SENSITIVE_FIELDS.has(name) ? "" : ((rawValues[name] ?? "") as string)]),
      ),
    [visibleFields, rawValues],
  );

  const schema = useMemo(() => buildSchema(visibleFields), [visibleFields]);
  const form = useZodForm(schema, { values: seededValues });

  const handleSubmit = (formValues: VaultFormValues) => {
    const config: Record<string, string> = Object.fromEntries(
      Object.entries(formValues).flatMap(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") return [[key, value]];
        if (!SENSITIVE_FIELDS.has(key)) return [[key, ""]];
        return [];
      }),
    );

    mutate(config, {
      onSuccess: () => {
        toast.success(t("adminPanel.hashicorpVault.toasts.updated"));
        onSuccess();
      },
      onError: (err) => {
        toast.fromError(err);
      },
    });
  };

  const handleCancel = () => {
    form.reset(seededValues);
    onCancel();
  };

  const renderField = (fieldName: string) => {
    const fieldSchema = properties[fieldName];
    if (!fieldSchema) return null;

    const isSensitive = SENSITIVE_FIELDS.has(fieldName);
    const existingValue = rawValues[fieldName];
    const hasExistingValue = isSensitive && existingValue != null && existingValue !== "";
    const placeholder = hasExistingValue
      ? t("adminPanel.hashicorpVault.editModal.keepExistingPlaceholder", { value: existingValue })
      : fieldSchema?.description;

    return (
      <FormField
        key={fieldName}
        control={form.control}
        name={fieldName}
        label={(FIELD_LABELS[fieldName] && t(FIELD_LABELS[fieldName])) ?? fieldName}
      >
        {({ ref, ...field }) =>
          isSensitive ? (
            <PasswordInput ref={ref} placeholder={placeholder} {...field} />
          ) : (
            <Input ref={ref} placeholder={fieldSchema?.description} {...field} />
          )
        }
      </FormField>
    );
  };

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[700px]">
        <DialogHeader>
          <DialogTitle>{t("adminPanel.hashicorpVault.editModal.title")}</DialogTitle>
        </DialogHeader>
        <form onSubmit={form.handleSubmit(handleSubmit)}>
          {FIELD_GROUPS.map((group, index) => (
            <div key={group.title}>
              {index > 0 && <Separator className="my-6" />}
              <h5 className="mb-1 text-base font-semibold text-foreground">{t(group.title)}</h5>
              {group.subtitle && <p className="mb-4 text-sm text-muted-foreground">{t(group.subtitle)}</p>}
              <FieldGroup>{group.fields.map(renderField)}</FieldGroup>
            </div>
          ))}
        </form>
        <DialogFooter>
          <div className="flex items-center justify-end gap-2">
            <Button type="button" variant="outline" onClick={handleCancel} disabled={isPending}>
              {t("adminPanel.hashicorpVault.editModal.cancel")}
            </Button>
            <Button type="button" disabled={isPending} onClick={() => void form.handleSubmit(handleSubmit)()}>
              {isPending && <UiLoadingSpinner className="size-4 mr-1" />}
              {isPending
                ? t("adminPanel.hashicorpVault.editModal.saving")
                : t("adminPanel.hashicorpVault.editModal.save")}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditHashicorpVaultModal;
