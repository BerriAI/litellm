import { TriangleAlert } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { z } from "zod";
import i18n from "@/i18n/i18n";
import { modelPatchUpdateCall } from "./networking";
import { toast } from "@/lib/toast";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Alert, AlertTitle } from "@/components/shared/Alert";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";

const updateCredentialsSchema = z.object({
  api_key: z.string().min(1, { error: () => i18n.t("updateModelCredentialsModal.apiKeyRequired") }),
});

type UpdateCredentialsValues = z.infer<typeof updateCredentialsSchema>;

const EMPTY_VALUES: UpdateCredentialsValues = { api_key: "" };

interface UpdateModelCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  accessToken: string;
  modelId: string;
  onUpdated: () => void;
}

export default function UpdateModelCredentialsModal({
  open,
  onCancel,
  accessToken,
  modelId,
  onUpdated,
}: UpdateModelCredentialsModalProps) {
  const { t } = useTranslation();
  const form = useZodForm(updateCredentialsSchema, { defaultValues: EMPTY_VALUES });
  const [isSaving, setIsSaving] = useState(false);

  const close = () => {
    form.reset(EMPTY_VALUES);
    onCancel();
  };

  const handleSubmit = async (values: UpdateCredentialsValues) => {
    const apiKey = values.api_key?.trim();
    if (!apiKey) {
      toast.fromError(t("updateModelCredentialsModal.apiKeyRequired"));
      return;
    }
    setIsSaving(true);
    try {
      await modelPatchUpdateCall(
        accessToken,
        { litellm_params: { api_key: apiKey }, model_info: { id: modelId } },
        modelId,
      );
      toast.success(t("updateModelCredentialsModal.updated"));
      form.reset(EMPTY_VALUES);
      onUpdated();
      onCancel();
    } catch (error) {
      console.error("Error updating API key:", error);
      toast.fromError(error);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => !nextOpen && close()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[520px]">
        <DialogHeader>
          <DialogTitle>{t("updateModelCredentialsModal.title")}</DialogTitle>
        </DialogHeader>
        <span className="block mb-4 text-sm text-muted-foreground">{t("updateModelCredentialsModal.description")}</span>
        <Alert variant="warning" className="mb-4">
          <TriangleAlert />
          <AlertTitle>{t("updateModelCredentialsModal.apiKeyOnlyWarning")}</AlertTitle>
        </Alert>
        <form onSubmit={form.handleSubmit(handleSubmit)}>
          <FieldGroup>
            <FormField control={form.control} name="api_key" label={t("updateModelCredentialsModal.newApiKey")}>
              {({ ref, ...field }) => (
                <PasswordInput
                  {...field}
                  ref={ref}
                  placeholder={t("updateModelCredentialsModal.newApiKeyPlaceholder")}
                  autoComplete="new-password"
                />
              )}
            </FormField>
          </FieldGroup>
          <div className="flex justify-end items-center mt-4 gap-2.5">
            <Button type="button" variant="outline" onClick={close}>
              {t("updateModelCredentialsModal.cancel")}
            </Button>
            <Button type="submit" disabled={isSaving}>
              {isSaving && <UiLoadingSpinner className="size-4" />}
              {t("updateModelCredentialsModal.submit")}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
