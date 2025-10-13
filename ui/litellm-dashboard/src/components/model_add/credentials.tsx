import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Card,
  Text,
  Badge,
  Button,
} from "@tremor/react";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { UploadProps } from "antd/es/upload";
import {
  credentialCreateCall,
  credentialDeleteCall,
  credentialUpdateCall,
  CredentialItem,
} from "@/components/networking"; // Assume this is your networking function
import AddCredentialsTab from "./add_credentials_tab";
import CredentialDeleteModal from "./CredentialDeleteModal";
import { Form } from "antd";
import NotificationsManager from "../molecules/notifications_manager";
interface CredentialsPanelProps {
  accessToken: string | null;
  uploadProps: UploadProps;
  credentialList: CredentialItem[];
  fetchCredentials: (accessToken: string) => Promise<void>;
}

const CredentialsPanel: React.FC<CredentialsPanelProps> = ({
  accessToken,
  uploadProps,
  credentialList,
  fetchCredentials,
}) => {
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [selectedCredential, setSelectedCredential] = useState<CredentialItem | null>(null);
  const [credentialToDelete, setCredentialToDelete] = useState<string | null>(null);
  const [form] = Form.useForm();

  const restrictedFields = ["credential_name", "custom_llm_provider"];
  const handleUpdateCredential = async (values: any) => {
    if (!accessToken) {
      return;
    }

    const filter_credential_values = Object.entries(values)
      .filter(([key]) => !restrictedFields.includes(key))
      .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
    // Transform form values into credential structure
    const newCredential = {
      credential_name: values.credential_name,
      credential_values: filter_credential_values,
      credential_info: {
        custom_llm_provider: values.custom_llm_provider,
      },
    };

    const response = await credentialUpdateCall(accessToken, values.credential_name, newCredential);
    NotificationsManager.success("Credential updated successfully");
    setIsUpdateModalOpen(false);
    fetchCredentials(accessToken);
  };

  const handleAddCredential = async (values: any) => {
    if (!accessToken) {
      return;
    }

    const filter_credential_values = Object.entries(values)
      .filter(([key]) => !restrictedFields.includes(key))
      .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
    // Transform form values into credential structure
    const newCredential = {
      credential_name: values.credential_name,
      credential_values: filter_credential_values,
      credential_info: {
        custom_llm_provider: values.custom_llm_provider,
      },
    };

    // Add to list and close modal
    const response = await credentialCreateCall(accessToken, newCredential);
    NotificationsManager.success("Credential added successfully");
    setIsAddModalOpen(false);
    fetchCredentials(accessToken);
  };

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    fetchCredentials(accessToken);
  }, [accessToken]);

  const renderProviderBadge = (provider: string) => {
    const providerColors: Record<string, string> = {
      openai: "blue",
      azure: "indigo",
      anthropic: "purple",
      default: "gray",
    };

    const color = providerColors[provider.toLowerCase()] || providerColors["default"];
    return (
      <Badge color={color as any} size="xs">
        {provider}
      </Badge>
    );
  };

  const handleDeleteCredential = async (credentialName: string) => {
    if (!accessToken) {
      return;
    }
    const response = await credentialDeleteCall(accessToken, credentialName);
    NotificationsManager.success("Credential deleted successfully");
    setCredentialToDelete(null);
    fetchCredentials(accessToken);
  };

  const openDeleteModal = (credentialName: string) => {
    setCredentialToDelete(credentialName);
  };

  const closeDeleteModal = () => {
    setCredentialToDelete(null);
  };

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      <div className="flex justify-between items-center mb-4">
        <Text>Configured credentials for different AI providers. Add and manage your API credentials.</Text>
      </div>

      <Card>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Credential Name</TableHeaderCell>
              <TableHeaderCell>Provider</TableHeaderCell>
              <TableHeaderCell>Description</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {!credentialList || credentialList.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="text-center py-4 text-gray-500">
                  No credentials configured
                </TableCell>
              </TableRow>
            ) : (
              credentialList.map((credential: CredentialItem, index: number) => (
                <TableRow key={index}>
                  <TableCell>{credential.credential_name}</TableCell>
                  <TableCell>
                    {renderProviderBadge((credential.credential_info?.custom_llm_provider as string) || "-")}
                  </TableCell>
                  <TableCell>{credential.credential_info?.description || "-"}</TableCell>
                  <TableCell>
                    <Button
                      icon={PencilAltIcon}
                      variant="light"
                      size="sm"
                      onClick={() => {
                        setSelectedCredential(credential);
                        setIsUpdateModalOpen(true);
                      }}
                    />
                    <Button
                      icon={TrashIcon}
                      variant="light"
                      size="sm"
                      onClick={() => openDeleteModal(credential.credential_name)}
                    />
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>
      <Button onClick={() => setIsAddModalOpen(true)} className="mt-4">
        Add Credential
      </Button>

      {isAddModalOpen && (
        <AddCredentialsTab
          onAddCredential={handleAddCredential}
          isVisible={isAddModalOpen}
          onCancel={() => setIsAddModalOpen(false)}
          uploadProps={uploadProps}
          addOrEdit="add"
          onUpdateCredential={handleUpdateCredential}
          existingCredential={null}
        />
      )}
      {isUpdateModalOpen && (
        <AddCredentialsTab
          onAddCredential={handleAddCredential}
          isVisible={isUpdateModalOpen}
          existingCredential={selectedCredential}
          onUpdateCredential={handleUpdateCredential}
          uploadProps={uploadProps}
          onCancel={() => setIsUpdateModalOpen(false)}
          addOrEdit="edit"
        />
      )}

      {credentialToDelete && (
        <CredentialDeleteModal
          isVisible={true}
          onCancel={closeDeleteModal}
          onConfirm={() => handleDeleteCredential(credentialToDelete)}
          credentialName={credentialToDelete}
        />
      )}
    </div>
  );
};

export default CredentialsPanel;
