"use client";

import { useTerminology } from "@/contexts/TerminologyContext";
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
import { extractProxyErrorMessage } from "@/lib/http/client";
import { stripMaskedSecrets } from "@/utils/maskedSecretUtils";
import { isProxyAdminRole } from "@/utils/roles";

import DeleteResourceModal from "../common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import CredentialModal from "./CredentialModal";
import CredentialsTable from "./CredentialsTable";
import { buildCredential, type CredentialSubmission } from "./credential_form_helpers";

export default function CredentialsPanel() {
  const { t } = useTranslation();
  const term = useTerminology();
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

  const handleUpdateCredential = async (submission: CredentialSubmission, valuesToDelete: readonly string[] = []) => {
    if (!accessToken) {
      return;
    }
    try {
      const newCredential = buildCredential(submission, stripMaskedSecrets(submission.credential_values));
      await credentialUpdateCall(accessToken, submission.credential_name, {
        ...newCredential,
        ...(valuesToDelete.length > 0 ? { credential_values_to_delete: valuesToDelete } : {}),
      });
      toast.success(t("modelAdd.credentialsPanel.updated"));
      setIsUpdateModalOpen(false);
      await refetchCredentials();
    } catch (error) {
      toast.error(extractProxyErrorMessage(error));
    }
  };

  const handleAddCredential = async (submission: CredentialSubmission) => {
    if (!accessToken) {
      return;
    }
    try {
      const newCredential = buildCredential(submission, submission.credential_values);
      await credentialCreateCall(accessToken, newCredential);
      toast.success(t("modelAdd.credentialsPanel.added"));
      setIsAddModalOpen(false);
      await refetchCredentials();
    } catch (error) {
      toast.error(extractProxyErrorMessage(error));
    }
  };

  const handleDeleteCredential = async () => {
    if (!accessToken || !credentialToDelete) {
      return;
    }
    setIsCredentialDeleting(true);
    try {
      await credentialDeleteCall(accessToken, credentialToDelete.credential_name);
      toast.success(t("modelAdd.credentialsPanel.deleted"));
      await refetchCredentials();
    } catch (error) {
      toast.error(t("modelAdd.credentialsPanel.deleteFailed"));
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
        <p className="text-sm text-muted-foreground">{term(t("modelAdd.credentialsPanel.description"))}</p>
        {canModifyCredentials && (
          <Button onClick={() => setIsAddModalOpen(true)}>
            <Plus className="size-4" />
            {t("modelAdd.credentialsPanel.addCredential")}
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
        />
      )}
      {isUpdateModalOpen && (
        <CredentialModal
          mode="edit"
          open={isUpdateModalOpen}
          existingCredential={selectedCredential}
          onSubmit={handleUpdateCredential}
          onCancel={() => setIsUpdateModalOpen(false)}
        />
      )}

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        onCancel={closeDeleteModal}
        onOk={handleDeleteCredential}
        title={t("modelAdd.credentialsPanel.deleteTitle")}
        message={t("modelAdd.credentialsPanel.deleteMessage")}
        resourceInformationTitle={t("modelAdd.credentialsPanel.deleteInformationTitle")}
        resourceInformation={[
          { label: t("modelAdd.credentialsPanel.credentialName"), value: credentialToDelete?.credential_name },
          {
            label: t("modelAdd.credentialsPanel.provider"),
            value: credentialToDelete?.credential_info?.custom_llm_provider || "-",
          },
        ]}
        confirmLoading={isCredentialDeleting}
        requiredConfirmation={credentialToDelete?.credential_name}
      />
    </div>
  );
}
