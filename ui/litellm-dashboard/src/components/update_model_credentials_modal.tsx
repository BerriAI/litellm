import { useLanguage } from "@/contexts/LanguageContext";
import { TriangleAlert } from "lucide-react";
import { useState } from "react";
import { z } from "zod";
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
import TranslatedMessage from "@/components/shared/TranslatedMessage";

// 校验保存固定原文，字段显示时再翻译，避免语言变化重置草稿或已有错误。
const updateCredentialsSchema = z.object({
  api_key: z.string().min(1, "Enter a new API key"),
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
  const { t } = useLanguage();
  const form = useZodForm(updateCredentialsSchema, { defaultValues: EMPTY_VALUES });
  const [isSaving, setIsSaving] = useState(false);

  const close = () => {
    form.reset(EMPTY_VALUES);
    onCancel();
  };

  const handleSubmit = async (values: UpdateCredentialsValues) => {
    const apiKey = values.api_key?.trim();
    if (!apiKey) {
      toast.fromError("Enter a new API key", { description: <TranslatedMessage message="Enter a new API key" /> });
      return;
    }
    setIsSaving(true);
    try {
      await modelPatchUpdateCall(
        accessToken,
        { litellm_params: { api_key: apiKey }, model_info: { id: modelId } },
        modelId,
      );
      toast.success(<TranslatedMessage message="API key updated" />);
      form.reset(EMPTY_VALUES);
      onUpdated();
      onCancel();
    } catch (error) {
      console.error("Error updating API key:", error);
      // 继续只显示固定失败提示，不将可能含密钥的服务端详情放到界面。
      toast.fromError("Failed to update API key", {
        description: <TranslatedMessage message="Failed to update API key" />,
      });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={(nextOpen) => !nextOpen && close()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[520px]">
        <DialogHeader>
          <DialogTitle>{t("Update API Key")}</DialogTitle>
        </DialogHeader>
        <span className="block mb-4 text-sm text-muted-foreground">
          {t(
            "Update this model's API key. Only the new key is sent; the rest of the deployment configuration is left untouched.",
          )}
        </span>
        <Alert variant="warning" className="mb-4">
          <TriangleAlert />
          <AlertTitle>
            {t(
              "Only the API key is rotated here. Models that authenticate with an Azure AD token, AWS credentials, or a Vertex service-account JSON aren't supported yet; update those from the model's LiteLLM Params for now.",
            )}
          </AlertTitle>
        </Alert>
        <form onSubmit={form.handleSubmit(handleSubmit)}>
          <FieldGroup>
            <FormField
              control={form.control}
              name="api_key"
              label={t("New API Key")}
              renderError={(error) =>
                error.message === "Enter a new API key" ? t("Enter a new API key") : error.message
              }
            >
              {({ ref, ...field }) => (
                <PasswordInput
                  {...field}
                  ref={ref}
                  placeholder={t("Enter the new API key")}
                  autoComplete="new-password"
                />
              )}
            </FormField>
          </FieldGroup>
          <div className="flex justify-end items-center mt-4 gap-2.5">
            <Button type="button" variant="outline" onClick={close}>
              {t("Cancel")}
            </Button>
            <Button type="submit" disabled={isSaving}>
              {isSaving && <UiLoadingSpinner className="size-4" />}
              {t("Update API Key")}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}
