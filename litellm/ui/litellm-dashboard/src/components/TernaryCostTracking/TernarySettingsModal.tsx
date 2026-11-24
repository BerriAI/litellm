import { useEffect, useMemo } from "react";
import { useTranslation } from "react-i18next";

import {
  useTernaryCreate,
  useTernaryUpdate,
  type TernarySettingsView,
} from "@/app/(dashboard)/hooks/ternary/useTernary";
import { FormField } from "@/components/shared/form/FormField";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";

import {
  buildCreatePayload,
  buildUpdatePayload,
  createSchema,
  EMPTY_TERNARY_FORM_VALUES,
  formValuesFromSettings,
  FREQUENCY_ITEMS,
  updateSchemaFor,
  type TernaryFormValues,
} from "./ternaryPayload";

interface TernarySettingsModalProps {
  open: boolean;
  settings: TernarySettingsView | null;
  onSaved: (message: string) => void;
  onCancel: () => void;
}

export default function TernarySettingsModal({ open, settings, onSaved, onCancel }: TernarySettingsModalProps) {
  const { t } = useTranslation();
  const isEdit = settings !== null;
  const currentBaseUrl = settings?.ternary_base_url ?? "";
  const schema = useMemo(() => (isEdit ? updateSchemaFor(currentBaseUrl) : createSchema), [isEdit, currentBaseUrl]);
  const form = useZodForm(schema, { defaultValues: EMPTY_TERNARY_FORM_VALUES });
  const createMutation = useTernaryCreate();
  const updateMutation = useTernaryUpdate();
  const isPending = createMutation.isPending || updateMutation.isPending;
  const frequency = form.watch("frequency");
  const frequencyItems = FREQUENCY_ITEMS.map((item) => ({ value: item.value, label: t(item.labelKey) }));
  const submitLabel = isEdit ? t("loggingAndAlerts.ternary.form.update") : t("loggingAndAlerts.ternary.form.create");

  useEffect(() => {
    if (open) {
      form.reset(settings ? formValuesFromSettings(settings) : EMPTY_TERNARY_FORM_VALUES);
    }
  }, [open, settings, form]);

  const callbacks = {
    onSuccess: (response: { message: string }) => onSaved(response.message),
    onError: (error: Error) => toast.error(error.message || t("loggingAndAlerts.ternary.form.saveFailed")),
  };

  const handleSubmit = (values: TernaryFormValues) => {
    if (isEdit) {
      updateMutation.mutate(buildUpdatePayload(values), callbacks);
    } else {
      createMutation.mutate(buildCreatePayload(values), callbacks);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {isEdit ? t("loggingAndAlerts.ternary.form.editTitle") : t("loggingAndAlerts.ternary.form.addTitle")}
          </DialogTitle>
          <DialogDescription>{t("loggingAndAlerts.ternary.form.description")}</DialogDescription>
        </DialogHeader>
        <form onSubmit={(event) => event.preventDefault()} noValidate>
          <FieldGroup>
            <FormField
              control={form.control}
              name="api_key"
              label={t("loggingAndAlerts.ternary.form.apiKey")}
              description={isEdit ? t("loggingAndAlerts.ternary.form.apiKeyHint") : undefined}
            >
              {({ ref, ...field }) => (
                <PasswordInput
                  {...field}
                  ref={ref}
                  autoComplete="off"
                  placeholder={
                    isEdit
                      ? t("loggingAndAlerts.ternary.form.newApiKeyPlaceholder")
                      : t("loggingAndAlerts.ternary.form.apiKeyPlaceholder")
                  }
                />
              )}
            </FormField>
            <FormField
              control={form.control}
              name="connection_id"
              label={t("loggingAndAlerts.ternary.form.connectionId")}
            >
              {({ ref, ...field }) => (
                <Input {...field} ref={ref} placeholder={t("loggingAndAlerts.ternary.form.connectionIdPlaceholder")} />
              )}
            </FormField>
            <FormField
              control={form.control}
              name="ternary_base_url"
              label={t("loggingAndAlerts.ternary.form.baseUrl")}
            >
              {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="https://api.ternary.app" />}
            </FormField>
            <FormField control={form.control} name="frequency" label={t("loggingAndAlerts.ternary.form.frequency")}>
              {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": describedBy }) => (
                <Select items={frequencyItems} value={value} onValueChange={onChange}>
                  <SelectTrigger id={id} aria-invalid={ariaInvalid} aria-describedby={describedBy} className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {frequencyItems.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </FormField>
            {frequency === "interval" && (
              <FormField
                control={form.control}
                name="interval_seconds"
                label={t("loggingAndAlerts.ternary.form.intervalSeconds")}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} inputMode="numeric" placeholder="300" />}
              </FormField>
            )}
          </FieldGroup>
        </form>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isPending}>
            {t("loggingAndAlerts.ternary.form.cancel")}
          </Button>
          <Button onClick={() => void form.handleSubmit(handleSubmit)()} disabled={isPending} aria-busy={isPending}>
            {isPending ? t("loggingAndAlerts.ternary.form.saving") : submitLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
