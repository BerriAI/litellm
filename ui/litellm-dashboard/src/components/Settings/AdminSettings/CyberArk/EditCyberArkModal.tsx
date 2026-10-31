"use client";

import { useCyberArkConfig } from "@/app/(dashboard)/hooks/configOverrides/useCyberArkConfig";
import { useUpdateCyberArkConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateCyberArkConfig";
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

interface CyberArkFieldGroup {
  title: ParseKeys;
  subtitle?: ParseKeys;
  fields: string[];
}

const FIELD_GROUPS: CyberArkFieldGroup[] = [
  {
    title: "adminPanel.cyberArk.editModal.groups.connection.title",
    fields: ["cyberark_api_base", "cyberark_account", "cyberark_username"],
  },
  {
    title: "adminPanel.cyberArk.editModal.groups.apiKey.title",
    subtitle: "adminPanel.cyberArk.editModal.groups.apiKey.subtitle",
    fields: ["cyberark_api_key"],
  },
  {
    title: "adminPanel.cyberArk.editModal.groups.certificate.title",
    subtitle: "adminPanel.cyberArk.editModal.groups.certificate.subtitle",
    fields: ["client_cert", "client_key"],
  },
  {
    title: "adminPanel.cyberArk.editModal.groups.advanced.title",
    subtitle: "adminPanel.cyberArk.editModal.groups.advanced.subtitle",
    fields: ["ssl_verify", "refresh_interval"],
  },
];

type CyberArkFormValues = Record<string, string>;

const buildSchema = (fields: readonly string[]): z.ZodType<CyberArkFormValues, CyberArkFormValues> =>
  z.object(
    Object.fromEntries(
      fields.map((name) => [
        name,
        name === "cyberark_api_base"
          ? z.string().refine((value) => value.length === 0 || /^https?:\/\/.+/.test(value), {
              error: () => i18n.t("adminPanel.cyberArk.editModal.errors.invalidUrl"),
            })
          : z.string(),
      ]),
    ),
  ) as unknown as z.ZodType<CyberArkFormValues, CyberArkFormValues>;

interface EditCyberArkModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const EditCyberArkModal: React.FC<EditCyberArkModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const { data } = useCyberArkConfig();
  const { mutate, isPending } = useUpdateCyberArkConfig(accessToken);

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

  const handleSubmit = (formValues: CyberArkFormValues) => {
    const config: Record<string, string> = Object.fromEntries(
      Object.entries(formValues).flatMap(([key, value]) => {
        if (value !== undefined && value !== null && value !== "") return [[key, value]];
        if (!SENSITIVE_FIELDS.has(key)) return [[key, ""]];
        return [];
      }),
    );

    mutate(config, {
      onSuccess: () => {
        toast.success(t("adminPanel.cyberArk.toasts.updated"));
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
      ? t("adminPanel.cyberArk.editModal.keepExistingPlaceholder", { value: existingValue })
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
          <DialogTitle>{t("adminPanel.cyberArk.editModal.title")}</DialogTitle>
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
              {t("adminPanel.cyberArk.editModal.cancel")}
            </Button>
            <Button type="button" disabled={isPending} onClick={() => void form.handleSubmit(handleSubmit)()}>
              {isPending && <UiLoadingSpinner className="size-4 mr-1" />}
              {isPending ? t("adminPanel.cyberArk.editModal.saving") : t("adminPanel.cyberArk.editModal.save")}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditCyberArkModal;
