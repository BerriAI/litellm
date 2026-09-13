"use client";

import { toast } from "@/lib/toast";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import React from "react";
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
  const form = useSSOSettingsForm("sso-settings");
  const { mutateAsync, isPending } = useEditSSOSettings();

  const handleFormSubmit = async (formValues: SSOSettingsFormValues) => {
    const payload = processSSOSettingsPayload(formValues);

    await mutateAsync(payload, {
      onSuccess: () => {
        toast.success("SSO settings added successfully");
        onSuccess();
      },
      onError: (error) => {
        toast.fromError("Failed to save SSO settings: " + parseErrorMessage(error));
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
          <DialogTitle>Add SSO</DialogTitle>
        </DialogHeader>
        <BaseSSOSettingsForm form={form} onFormSubmit={handleFormSubmit} />
        <DialogFooter>
          <div className="flex items-center justify-end gap-2">
            <Button type="button" variant="outline" onClick={handleCancel} disabled={isPending}>
              Cancel
            </Button>
            <Button
              type="button"
              disabled={isPending}
              onClick={submitMountedSSOValues(form, "sso-settings", handleFormSubmit)}
            >
              {isPending && <UiLoadingSpinner className="size-4 mr-1" />}
              {isPending ? "Adding..." : "Add SSO"}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default AddSSOSettingsModal;
