"use client";

import { toast } from "@/lib/toast";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import React from "react";
import { useTranslation } from "react-i18next";
import BaseSSOSettingsForm, {
  emptySSOSettingsFormValues,
  submitMountedSSOValues,
  useSSOSettingsForm,
  type SSOSettingsFormValues,
} from "./BaseSSOSettingsForm";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import { processSSOSettingsPayload } from "../utils";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface AddSSOSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const AddSSOSettingsModal: React.FC<AddSSOSettingsModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { t } = useTranslation();
  const form = useSSOSettingsForm("sso-settings");
  const { mutateAsync, isPending } = useEditSSOSettings();

  const handleFormSubmit = async (formValues: SSOSettingsFormValues) => {
    const payload = processSSOSettingsPayload(formValues);

    await mutateAsync(payload, {
      onSuccess: () => {
        toast.success(t("adminPanel.ssoSettings.addModal.toasts.success"));
        onSuccess();
      },
      onError: (error) => {
        toast.fromError(t("adminPanel.ssoSettings.addModal.toasts.saveFailed", { error: parseErrorMessage(error) }));
      },
    });
  };

  const handleCancel = () => {
    form.reset(emptySSOSettingsFormValues);
    onCancel();
  };

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{t("adminPanel.ssoSettings.addModal.title")}</DialogTitle>
        </DialogHeader>
        <BaseSSOSettingsForm form={form} onFormSubmit={handleFormSubmit} />
        <DialogFooter>
          <div className="flex items-center justify-end gap-2">
            <Button type="button" variant="outline" onClick={handleCancel} disabled={isPending}>
              {t("adminPanel.ssoSettings.addModal.cancel")}
            </Button>
            <Button
              type="button"
              disabled={isPending}
              onClick={submitMountedSSOValues(form, "sso-settings", handleFormSubmit)}
            >
              {isPending && <UiLoadingSpinner className="size-4 mr-1" />}
              {isPending
                ? t("adminPanel.ssoSettings.addModal.submitting")
                : t("adminPanel.ssoSettings.addModal.submit")}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default AddSSOSettingsModal;
