import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import { useSSOSettings } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import React from "react";
import DeleteResourceModal from "../../../../common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import { parseErrorMessage } from "../../../../shared/errorUtils";
import { CLEARED_SSO_SETTINGS, detectSSOProvider } from "../utils";

interface DeleteSSOSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const DeleteSSOSettingsModal: React.FC<DeleteSSOSettingsModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { data: ssoSettings } = useSSOSettings();
  const { mutateAsync: editSSOSettings, isPending: isEditingSSOSettings } = useEditSSOSettings();

  // Handle clearing SSO settings
  const handleClearSSO = async () => {
    await editSSOSettings(CLEARED_SSO_SETTINGS, {
      onSuccess: () => {
        toast.success("SSO settings cleared successfully");
        onCancel();
        onSuccess();
      },
      onError: (error) => {
        toast.fromError("Failed to clear SSO settings: " + parseErrorMessage(error));
      },
    });
  };

  return (
    <DeleteResourceModal
      isOpen={isVisible}
      title="Confirm Clear SSO Settings"
      alertMessage="This action cannot be undone."
      message="Are you sure you want to clear all SSO settings? Users will no longer be able to login using SSO after this change."
      resourceInformationTitle="SSO Settings"
      resourceInformation={[
        { label: "Provider", value: (ssoSettings?.values && detectSSOProvider(ssoSettings?.values)) || "Generic" },
      ]}
      onCancel={onCancel}
      onOk={handleClearSSO}
      confirmLoading={isEditingSSOSettings}
    />
  );
};

export default DeleteSSOSettingsModal;
