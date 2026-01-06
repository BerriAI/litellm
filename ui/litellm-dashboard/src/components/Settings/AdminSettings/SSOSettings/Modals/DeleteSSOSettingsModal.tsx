import { Modal } from "antd";
import React from "react";
import NotificationsManager from "../../../../molecules/notifications_manager";
import { updateSSOSettings } from "../../../../networking";
import { parseErrorMessage } from "../../../../shared/errorUtils";

interface DeleteSSOSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
  accessToken: string | null;
}

const DeleteSSOSettingsModal: React.FC<DeleteSSOSettingsModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  accessToken,
}) => {
  // Handle clearing SSO settings
  const handleClearSSO = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    try {
      // Clear all SSO settings
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
      };

      await updateSSOSettings(accessToken, clearSettings);

      NotificationsManager.success("SSO settings cleared successfully");

      // Close modal and trigger success callback
      onCancel();
      onSuccess();
    } catch (error) {
      console.error("Failed to clear SSO settings:", error);
      NotificationsManager.fromBackend("Failed to clear SSO settings: " + parseErrorMessage(error));
    }
  };

  return (
    <Modal
      title="Confirm Clear SSO Settings"
      visible={isVisible}
      onOk={handleClearSSO}
      onCancel={onCancel}
      okText="Yes, Clear"
      cancelText="Cancel"
      okButtonProps={{
        danger: true,
        style: {
          backgroundColor: "#dc2626",
          borderColor: "#dc2626",
        },
      }}
    >
      <p>Are you sure you want to clear all SSO settings? This action cannot be undone.</p>
      <p>Users will no longer be able to login using SSO after this change.</p>
    </Modal>
  );
};

export default DeleteSSOSettingsModal;
