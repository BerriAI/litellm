import { useLanguage } from "@/contexts/LanguageContext";
import { useEffect } from "react";
import { z } from "zod";

import { useCloudZeroCreate } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate";
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
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface CloudZeroCreationModalProps {
  open: boolean;
  onOk: () => void;
  onCancel: () => void;
}

// 校验保存本地原文，已有错误只在显示处翻译，语言变化不触发重置或校验。
const createSchema = z.object({
  api_key: z.string().min(1, "Please enter your CloudZero API key"),
  connection_id: z.string().min(1, "Please enter your CloudZero connection ID"),
  timezone: z.string(),
});

export default function CloudZeroCreationModal({ open, onOk, onCancel }: CloudZeroCreationModalProps) {
  const { t } = useLanguage();
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
        toast.success(<TranslatedMessage message="CloudZero integration created successfully" />);
        form.reset(EMPTY_CLOUDZERO_FORM_VALUES);
        onOk();
      },
      onError: (error: Error) => {
        toast.error(
          error.cause === LOCAL_CLOUDZERO_ERROR ? (
            <TranslatedMessage message={error.message} />
          ) : (
            error.message || <TranslatedMessage message="Failed to create CloudZero integration" />
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
          <DialogTitle>{t("Create CloudZero Integration")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={(event) => event.preventDefault()} noValidate>
            <FieldGroup>
              <FormField
                control={form.control}
                name="api_key"
                label={t("CloudZero API Key")}
                renderError={({ message, type }) =>
                  type === "too_small" && message === "Please enter your CloudZero API key" ? t(message) : message
                }
              >
                {({ ref, ...field }) => (
                  <CloudZeroApiKeyInput {...field} ref={ref} placeholder={t("Enter your CloudZero API key")} />
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
          <Button variant="outline" onClick={handleCancel} disabled={createMutation.isPending}>
            {t("Cancel")}
          </Button>
          <Button
            onClick={() => void form.handleSubmit(handleSubmit)()}
            disabled={createMutation.isPending}
            aria-busy={createMutation.isPending}
          >
            {createMutation.isPending ? t("Creating...") : t("Create")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
