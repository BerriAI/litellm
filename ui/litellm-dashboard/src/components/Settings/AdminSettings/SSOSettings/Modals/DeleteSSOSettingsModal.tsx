import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import { useSSOSettings } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import React from "react";
import { useTranslation } from "react-i18next";
import DeleteResourceModal from "../../../../common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import { parseErrorMessage } from "../../../../shared/errorUtils";
import { detectSSOProvider } from "../utils";

interface DeleteSSOSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const DeleteSSOSettingsModal: React.FC<DeleteSSOSettingsModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { t } = useTranslation();
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
      saml_idp_metadata_url: null,
      saml_idp_metadata_xml: null,
      saml_sp_entity_id: null,
      saml_allow_unsolicited: null,
      proxy_base_url: null,
      user_email: null,
      sso_provider: null,
      role_mappings: null,
      team_mappings: null,
    };

    await editSSOSettings(clearSettings, {
      onSuccess: () => {
        toast.success(t("adminPanel.ssoSettings.deleteModal.toasts.success"));
        onCancel();
        onSuccess();
      },
      onError: (error) => {
        toast.fromError(t("adminPanel.ssoSettings.deleteModal.toasts.failed", { error: parseErrorMessage(error) }));
      },
    });
  };

  return (
    <DeleteResourceModal
      isOpen={isVisible}
      title={t("adminPanel.ssoSettings.deleteModal.title")}
      alertMessage={t("adminPanel.ssoSettings.deleteModal.alert")}
      message={t("adminPanel.ssoSettings.deleteModal.message")}
      resourceInformationTitle={t("adminPanel.ssoSettings.deleteModal.resourceTitle")}
      resourceInformation={[
        {
          label: t("adminPanel.ssoSettings.deleteModal.provider"),
          value:
            (ssoSettings?.values && detectSSOProvider(ssoSettings?.values)) ||
            t("adminPanel.ssoSettings.deleteModal.providerFallback"),
        },
      ]}
      onCancel={onCancel}
      onOk={handleClearSSO}
      confirmLoading={isEditingSSOSettings}
    />
  );
};

export default DeleteSSOSettingsModal;
