import { useEffect } from "react";
import { useTranslation } from "react-i18next";
import { z } from "zod";

import { useCloudZeroCreate } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Input } from "@/components/ui/input";
import { TooltipProvider } from "@/components/ui/tooltip";
import i18n from "@/i18n/i18n";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";

import { CloudZeroApiKeyInput, labelWithHint } from "./CloudZeroFormControls";
import { buildCloudZeroPayload, EMPTY_CLOUDZERO_FORM_VALUES, type CloudZeroFormValues } from "./cloudZeroPayload";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface CloudZeroCreationModalProps {
  open: boolean;
  onOk: () => void;
  onCancel: () => void;
}

const createSchema = z.object({
  api_key: z.string().min(1, { error: () => i18n.t("loggingAndAlerts.cloudZero.form.apiKeyRequired") }),
  connection_id: z.string().min(1, { error: () => i18n.t("loggingAndAlerts.cloudZero.form.connectionIdRequired") }),
  timezone: z.string(),
});

export default function CloudZeroCreationModal({ open, onOk, onCancel }: CloudZeroCreationModalProps) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const form = useZodForm(createSchema, { defaultValues: EMPTY_CLOUDZERO_FORM_VALUES });
  const createMutation = useCloudZeroCreate(accessToken || "");

  useEffect(() => {
    if (open) {
      form.reset(EMPTY_CLOUDZERO_FORM_VALUES);
    }
  }, [open, form]);

  const handleSubmit = (values: CloudZeroFormValues) => {
    createMutation.mutate(buildCloudZeroPayload(values), {
      onSuccess: () => {
        toast.success(t("loggingAndAlerts.cloudZero.createModal.toasts.created"));
        form.reset(EMPTY_CLOUDZERO_FORM_VALUES);
        onOk();
      },
      onError: (error: Error) => {
        toast.error(error.message || t("loggingAndAlerts.cloudZero.createModal.toasts.createFailed"));
      },
    });
  };

  const handleCancel = () => {
    form.reset(EMPTY_CLOUDZERO_FORM_VALUES);
    onCancel();
  };

  return (
    <Dialog open={open} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{t("loggingAndAlerts.cloudZero.createModal.title")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={(event) => event.preventDefault()} noValidate>
            <FieldGroup>
              <FormField control={form.control} name="api_key" label={t("loggingAndAlerts.cloudZero.form.apiKey")}>
                {({ ref, ...field }) => (
                  <CloudZeroApiKeyInput
                    {...field}
                    ref={ref}
                    placeholder={t("loggingAndAlerts.cloudZero.form.apiKeyPlaceholder")}
                  />
                )}
              </FormField>
              <FormField
                control={form.control}
                name="connection_id"
                label={t("loggingAndAlerts.cloudZero.form.connectionId")}
              >
                {({ ref, ...field }) => (
                  <Input
                    {...field}
                    ref={ref}
                    placeholder={t("loggingAndAlerts.cloudZero.form.connectionIdPlaceholder")}
                  />
                )}
              </FormField>
              <FormField
                control={form.control}
                name="timezone"
                label={labelWithHint(
                  t("loggingAndAlerts.cloudZero.form.timezone"),
                  t("loggingAndAlerts.cloudZero.form.timezoneHint"),
                )}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="UTC" />}
              </FormField>
            </FieldGroup>
          </form>
        </TooltipProvider>
        <DialogFooter>
          <Button variant="outline" onClick={handleCancel} disabled={createMutation.isPending}>
            {t("loggingAndAlerts.cloudZero.createModal.cancel")}
          </Button>
          <Button
            onClick={() => void form.handleSubmit(handleSubmit)()}
            disabled={createMutation.isPending}
            aria-busy={createMutation.isPending}
          >
            {createMutation.isPending
              ? t("loggingAndAlerts.cloudZero.createModal.submitting")
              : t("loggingAndAlerts.cloudZero.createModal.submit")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
