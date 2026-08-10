"use client";

import { UploadProps } from "antd/es/upload";
import { Plus } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  credentialCreateCall,
  credentialDeleteCall,
  CredentialItem,
  credentialUpdateCall,
} from "@/components/networking";
import { Button } from "@/components/ui/button";
import { stripMaskedSecrets } from "@/utils/maskedSecretUtils";
import { isProxyAdminRole } from "@/utils/roles";

import DeleteResourceModal from "../common_components/DeleteResourceModal";
import NotificationsManager from "../molecules/notifications_manager";
import CredentialModal from "./CredentialModal";
import CredentialsTable from "./CredentialsTable";

interface CredentialsPanelProps {
  uploadProps: UploadProps;
}

const restrictedFields = ["credential_name", "custom_llm_provider"];

const buildCredential = (values: Record<string, unknown>, credentialValues: Record<string, unknown>) => ({
  credential_name: values.credential_name as string,
  credential_values: credentialValues,
  credential_info: {
    custom_llm_provider: values.custom_llm_provider as string,
  },
});

const withoutRestrictedFields = (values: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(values).filter(([key]) => !restrictedFields.includes(key)));

export default function CredentialsPanel({ uploadProps }: CredentialsPanelProps) {
  const { t } = useTranslation("gateway");
  const { accessToken, userRole } = useAuthorized();
  // Admin Viewer follows the read-parity rule: see credentials, do not modify.
  const canModifyCredentials = isProxyAdminRole(userRole ?? "");
  const { data: credentialsResponse, isLoading, refetch: refetchCredentials } = useCredentials();
  const credentialList = credentialsResponse?.credentials || [];

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [selectedCredential, setSelectedCredential] = useState<CredentialItem | null>(null);
  const [credentialToDelete, setCredentialToDelete] = useState<CredentialItem | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isCredentialDeleting, setIsCredentialDeleting] = useState(false);

  const handleUpdateCredential = async (values: Record<string, unknown>) => {
    if (!accessToken) {
      return;
    }
    try {
      const newCredential = buildCredential(values, stripMaskedSecrets(withoutRestrictedFields(values)));
      await credentialUpdateCall(accessToken, values.credential_name as string, newCredential);
      NotificationsManager.success(t("models.credentials.notifications.updated"));
      setIsUpdateModalOpen(false);
      await refetchCredentials();
    } catch (error) {
      NotificationsManager.error(t("models.credentials.notifications.updateFailed"));
    }
  };

  const handleAddCredential = async (values: Record<string, unknown>) => {
    if (!accessToken) {
      return;
    }
    try {
      const newCredential = buildCredential(values, withoutRestrictedFields(values));
      await credentialCreateCall(accessToken, newCredential);
      NotificationsManager.success(t("models.credentials.notifications.added"));
      setIsAddModalOpen(false);
      await refetchCredentials();
    } catch (error) {
      NotificationsManager.error(t("models.credentials.notifications.addFailed"));
    }
  };

  const handleDeleteCredential = async () => {
    if (!accessToken || !credentialToDelete) {
      return;
    }
    setIsCredentialDeleting(true);
    try {
      await credentialDeleteCall(accessToken, credentialToDelete.credential_name);
      NotificationsManager.success(t("models.credentials.notifications.deleted"));
      await refetchCredentials();
    } catch (error) {
      NotificationsManager.error(t("models.credentials.notifications.deleteFailed"));
    } finally {
      setCredentialToDelete(null);
      setIsDeleteModalOpen(false);
      setIsCredentialDeleting(false);
    }
  };

  const openEditModal = (credential: CredentialItem) => {
    setSelectedCredential(credential);
    setIsUpdateModalOpen(true);
  };

  const openDeleteModal = (credential: CredentialItem) => {
    setCredentialToDelete(credential);
    setIsDeleteModalOpen(true);
  };

  const closeDeleteModal = () => {
    setCredentialToDelete(null);
    setIsDeleteModalOpen(false);
  };

  return (
    <div className="mx-auto flex w-full flex-auto flex-col gap-4 overflow-y-auto p-2">
      <div className="flex items-center justify-between gap-4">
        <p className="text-sm text-muted-foreground">
          {t("models.credentials.description")}
        </p>
        {canModifyCredentials && (
          <Button onClick={() => setIsAddModalOpen(true)}>
            <Plus className="size-4" />
            {t("models.credentials.add")}
          </Button>
        )}
      </div>

      <CredentialsTable
        credentials={credentialList}
        canModifyCredentials={canModifyCredentials}
        onEdit={openEditModal}
        onDelete={openDeleteModal}
        isLoading={isLoading}
      />

      {isAddModalOpen && (
        <CredentialModal
          mode="add"
          onSubmit={handleAddCredential}
          open={isAddModalOpen}
          onCancel={() => setIsAddModalOpen(false)}
          uploadProps={uploadProps}
        />
      )}
      {isUpdateModalOpen && (
        <CredentialModal
          mode="edit"
          open={isUpdateModalOpen}
          existingCredential={selectedCredential}
          onSubmit={handleUpdateCredential}
          uploadProps={uploadProps}
          onCancel={() => setIsUpdateModalOpen(false)}
        />
      )}

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        onCancel={closeDeleteModal}
        onOk={handleDeleteCredential}
        title={t("models.credentials.deleteDialog.title")}
        message={t("models.credentials.deleteDialog.message")}
        resourceInformationTitle={t("models.credentials.deleteDialog.information")}
        resourceInformation={[
          { label: t("models.credentials.deleteDialog.name"), value: credentialToDelete?.credential_name },
          {
            label: t("models.credentials.deleteDialog.provider"),
            value: credentialToDelete?.credential_info?.custom_llm_provider || "-",
          },
        ]}
        confirmLoading={isCredentialDeleting}
        requiredConfirmation={credentialToDelete?.credential_name}
      />
    </div>
  );
}
