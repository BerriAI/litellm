import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import { useSSOSettings } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import React from "react";
import DeleteResourceModal from "../../../../common_components/DeleteResourceModal";
import NotificationsManager from "../../../../molecules/notifications_manager";
import { parseErrorMessage } from "../../../../shared/errorUtils";
import { detectSSOProvider } from "../utils";

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
    const clearSettings = {
      google_client_id: null,
      google_client_secret: null,
      microsoft_client_id: null,
      microsoft_client_secret: null,
      microsoft_tenant: null,
      generic_client_id: null,
      generic_client_secret: null,
      generic_authorization_endpoint: null,
      generic_token_endpoint: null,
      generic_userinfo_endpoint: null,
      proxy_base_url: null,
      user_email: null,
      sso_provider: null,
      role_mappings: null,
      team_mappings: null,
    };

    await editSSOSettings(clearSettings, {
      onSuccess: () => {
        NotificationsManager.success("SSO settings cleared successfully");
        onCancel();
        onSuccess();
      },
      onError: (error) => {
        NotificationsManager.fromBackend("Failed to clear SSO settings: " + parseErrorMessage(error));
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
