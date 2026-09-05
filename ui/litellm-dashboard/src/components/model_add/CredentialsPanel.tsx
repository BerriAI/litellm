"use client";

import { Plus } from "lucide-react";
import { useState } from "react";

import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import {
  credentialCreateCall,
  credentialDeleteCall,
  CredentialItem,
  credentialUpdateCall,
  discoverProviderModelsCall,
  getCredentialJwksCall,
  type ProviderModelDiscoveryRequest,
} from "@/components/networking";
import { Button } from "@/components/ui/button";
import { stripMaskedSecrets } from "@/utils/maskedSecretUtils";
import { isProxyAdminRole } from "@/utils/roles";

import DeleteResourceModal from "../common_components/DeleteResourceModal";
import { toast } from "@/lib/toast";
import CredentialModal from "./CredentialModal";
import CredentialsTable from "./CredentialsTable";
import { buildCredential, withoutRestrictedFields } from "./credential_form_helpers";

export default function CredentialsPanel() {
  const { accessToken, userRole } = useAuthorized();
  const testConnection = (request: ProviderModelDiscoveryRequest) =>
    accessToken ? discoverProviderModelsCall(accessToken, request) : Promise.reject(new Error("Not signed in"));
  const loadJwks = (credentialName: string) =>
    accessToken ? getCredentialJwksCall(accessToken, credentialName) : Promise.reject(new Error("Not signed in"));
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

  const handleUpdateCredential = async (
    values: Record<string, unknown>,
    credentialValuesToDelete: string[] = [],
  ): Promise<boolean> => {
    if (!accessToken) {
      return false;
    }
    try {
      const newCredential = {
        ...buildCredential(values, stripMaskedSecrets(withoutRestrictedFields(values))),
        ...(credentialValuesToDelete.length > 0 ? { credential_values_to_delete: credentialValuesToDelete } : {}),
      };
      await credentialUpdateCall(accessToken, values.credential_name as string, newCredential);
      toast.success("Credential updated successfully");
      setIsUpdateModalOpen(false);
      await refetchCredentials();
      return true;
    } catch (error) {
      toast.error("Failed to update credential");
      return false;
    }
  };

  const handleAddCredential = async (values: Record<string, unknown>): Promise<boolean> => {
    if (!accessToken) {
      return false;
    }
    try {
      const newCredential = buildCredential(values, withoutRestrictedFields(values));
      await credentialCreateCall(accessToken, newCredential);
      toast.success("Credential added successfully");
      setIsAddModalOpen(false);
      await refetchCredentials();
      return true;
    } catch (error) {
      toast.error("Failed to add credential");
      return false;
    }
  };

  const handleDeleteCredential = async () => {
    if (!accessToken || !credentialToDelete) {
      return;
    }
    setIsCredentialDeleting(true);
    try {
      await credentialDeleteCall(accessToken, credentialToDelete.credential_name);
      toast.success("Credential deleted successfully");
      await refetchCredentials();
    } catch (error) {
      toast.error("Failed to delete credential");
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
          Configured credentials for different AI providers. Add and manage your API credentials.
        </p>
        {canModifyCredentials && (
          <Button onClick={() => setIsAddModalOpen(true)}>
            <Plus className="size-4" />
            Add Credential
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
          testConnection={testConnection}
          loadJwks={loadJwks}
        />
      )}
      {isUpdateModalOpen && (
        <CredentialModal
          mode="edit"
          open={isUpdateModalOpen}
          existingCredential={selectedCredential}
          onSubmit={handleUpdateCredential}
          onCancel={() => setIsUpdateModalOpen(false)}
          testConnection={testConnection}
          loadJwks={loadJwks}
        />
      )}

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        onCancel={closeDeleteModal}
        onOk={handleDeleteCredential}
        title="Delete Credential?"
        message="Are you sure you want to delete this credential? This action cannot be undone and may break existing integrations."
        resourceInformationTitle="Credential Information"
        resourceInformation={[
          { label: "Credential Name", value: credentialToDelete?.credential_name },
          { label: "Provider", value: credentialToDelete?.credential_info?.custom_llm_provider || "-" },
        ]}
        confirmLoading={isCredentialDeleting}
        requiredConfirmation={credentialToDelete?.credential_name}
      />
    </div>
  );
}
