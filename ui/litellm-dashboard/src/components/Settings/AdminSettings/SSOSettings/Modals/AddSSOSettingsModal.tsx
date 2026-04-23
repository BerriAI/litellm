"use client";

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
import { Form } from "antd";
import React from "react";
import BaseSSOSettingsForm from "./BaseSSOSettingsForm";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import { processSSOSettingsPayload } from "../utils";

interface AddSSOSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const AddSSOSettingsModal: React.FC<AddSSOSettingsModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const { mutateAsync, isPending } = useEditSSOSettings();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleFormSubmit = async (formValues: Record<string, any>) => {
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
  };

  const handleCancel = () => {
    form.resetFields();
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
        <BaseSSOSettingsForm form={form} onFormSubmit={handleFormSubmit} />
        <DialogFooter>
          <Button variant="outline" onClick={handleCancel} disabled={isPending}>
            Cancel
          </Button>
          <Button onClick={() => form.submit()} disabled={isPending}>
            {isPending ? "Adding..." : "Add SSO"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default AddSSOSettingsModal;
