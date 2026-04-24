"use client";

import React from "react";
import { FormProvider, useForm } from "react-hook-form";

import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";

import BaseSSOSettingsForm, { SSOSettingsFormValues } from "./BaseSSOSettingsForm";
import { processSSOSettingsPayload } from "../utils";

interface AddSSOSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const emptyValues: SSOSettingsFormValues = {
  sso_provider: undefined,
  user_email: "",
  proxy_base_url: "",
};

const AddSSOSettingsModal: React.FC<AddSSOSettingsModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
}) => {
  const form = useForm<SSOSettingsFormValues>({
    defaultValues: emptyValues,
    mode: "onSubmit",
  });
  const { mutateAsync, isPending } = useEditSSOSettings();

  const onSubmit = form.handleSubmit(async (formValues) => {
    const payload = processSSOSettingsPayload(formValues);

    await mutateAsync(payload, {
      onSuccess: () => {
        NotificationsManager.success("SSO settings added successfully");
        onSuccess();
      },
      onError: (error) => {
        NotificationsManager.fromBackend(
          "Failed to save SSO settings: " + parseErrorMessage(error),
        );
      },
    });
  });

  const handleCancel = () => {
    form.reset(emptyValues);
    onCancel();
  };

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-[800px]">
        <DialogHeader>
          <DialogTitle>Add SSO</DialogTitle>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={onSubmit}>
            <BaseSSOSettingsForm />
            <DialogFooter className="mt-6">
              <Button
                type="button"
                variant="outline"
                onClick={handleCancel}
                disabled={isPending}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isPending}>
                {isPending ? "Adding..." : "Add SSO"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default AddSSOSettingsModal;
