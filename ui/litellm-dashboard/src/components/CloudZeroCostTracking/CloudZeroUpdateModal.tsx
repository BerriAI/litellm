import { useLanguage } from "@/contexts/LanguageContext";
import { useEffect } from "react";
import { z } from "zod";

import { useCloudZeroUpdateSettings } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings";
import { LOCAL_CLOUDZERO_ERROR } from "@/app/(dashboard)/hooks/cloudzero/cloudZeroErrors";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Input } from "@/components/ui/input";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import TranslatedMessage from "@/components/shared/TranslatedMessage";

import { CloudZeroApiKeyInput, labelWithHint } from "./CloudZeroFormControls";
import { buildCloudZeroPayload, EMPTY_CLOUDZERO_FORM_VALUES, type CloudZeroFormValues } from "./cloudZeroPayload";
import { CloudZeroSettings } from "./types";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface CloudZeroUpdateModalProps {
  open: boolean;
  onOk: () => void;
  onCancel: () => void;
  settings: CloudZeroSettings;
}

// 密钥仍可留空，仅在显示字段错误时翻译固定的本地必填提示。
const updateSchema = z.object({
  api_key: z.string(),
  connection_id: z.string().min(1, "Please enter your CloudZero connection ID"),
  timezone: z.string(),
});

export default function CloudZeroUpdateModal({ open, onOk, onCancel, settings }: CloudZeroUpdateModalProps) {
  const { t } = useLanguage();
  const { accessToken } = useAuthorized();
  const form = useZodForm(updateSchema, { defaultValues: EMPTY_CLOUDZERO_FORM_VALUES });
  const updateMutation = useCloudZeroUpdateSettings(accessToken || "");

  useEffect(() => {
    if (open && settings) {
      form.reset({
        connection_id: settings.connection_id ?? "",
        timezone: settings.timezone || "UTC",
        api_key: "",
      });
    } else if (open) {
      form.reset(EMPTY_CLOUDZERO_FORM_VALUES);
    }
  }, [open, settings, form]);

  const handleSubmit = (values: CloudZeroFormValues) => {
    updateMutation.mutate(buildCloudZeroPayload(values), {
      onSuccess: () => {
        toast.success(<TranslatedMessage message="CloudZero integration updated successfully" />);
        form.reset(EMPTY_CLOUDZERO_FORM_VALUES);
        onOk();
      },
      onError: (error: Error) => {
        toast.error(
          error.cause === LOCAL_CLOUDZERO_ERROR ? (
            <TranslatedMessage message={error.message} />
          ) : (
            error.message || <TranslatedMessage message="Failed to update CloudZero integration" />
          ),
        );
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
          <DialogTitle>{t("Edit CloudZero Integration")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={(event) => event.preventDefault()} noValidate>
            <FieldGroup>
              <FormField
                control={form.control}
                name="api_key"
                label={labelWithHint(t("CloudZero API Key"), t("Leave empty to keep the existing API key"))}
              >
                {({ ref, ...field }) => (
                  <CloudZeroApiKeyInput {...field} ref={ref} placeholder={t("Leave empty to keep existing")} />
                )}
              </FormField>
              <FormField
                control={form.control}
                name="connection_id"
                label={t("Connection ID")}
                renderError={({ message, type }) =>
                  type === "too_small" && message === "Please enter your CloudZero connection ID" ? t(message) : message
                }
              >
                {({ ref, ...field }) => (
                  <Input {...field} ref={ref} placeholder={t("Enter your CloudZero connection ID")} />
                )}
              </FormField>
              <FormField
                control={form.control}
                name="timezone"
                label={labelWithHint(t("Timezone"), t("Timezone for date handling (defaults to UTC if not provided)"))}
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder="UTC" />}
              </FormField>
            </FieldGroup>
          </form>
        </TooltipProvider>
        <DialogFooter>
          <Button variant="outline" onClick={handleCancel} disabled={updateMutation.isPending}>
            {t("Cancel")}
          </Button>
          <Button
            onClick={() => void form.handleSubmit(handleSubmit)()}
            disabled={updateMutation.isPending}
            aria-busy={updateMutation.isPending}
          >
            {updateMutation.isPending ? t("Updating...") : t("Update")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
