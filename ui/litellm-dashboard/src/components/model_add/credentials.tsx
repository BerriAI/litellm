import {
  credentialCreateCall,
  credentialDeleteCall,
  CredentialItem,
  credentialUpdateCall,
} from "@/components/networking"; // Assume this is your networking function
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import {
  Badge,
  Button,
  Card,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Text,
} from "@tremor/react";
import { Form } from "antd";
import { UploadProps } from "antd/es/upload";
import { useState } from "react";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import NotificationsManager from "../molecules/notifications_manager";
import AddCredentialsTab from "./AddCredentialModal";
import EditCredentialsModal from "./EditCredentialModal";
import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
interface CredentialsPanelProps {
  uploadProps: UploadProps;
}

const CredentialsPanel: React.FC<CredentialsPanelProps> = ({ uploadProps }) => {
  const { accessToken } = useAuthorized();
  const { data: credentialsResponse, refetch: refetchCredentials } = useCredentials();
  const credentialList = credentialsResponse?.credentials || [];

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [selectedCredential, setSelectedCredential] = useState<CredentialItem | null>(null);
  const [credentialToDelete, setCredentialToDelete] = useState<CredentialItem | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isCredentialDeleting, setIsCredentialDeleting] = useState(false);
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

    await credentialUpdateCall(accessToken, values.credential_name, newCredential);
    NotificationsManager.success("Credential updated successfully");
    setIsUpdateModalOpen(false);
    await refetchCredentials();
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
    await credentialCreateCall(accessToken, newCredential);
    NotificationsManager.success("Credential added successfully");
    setIsAddModalOpen(false);
    await refetchCredentials();
  };

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

  const handleDeleteCredential = async () => {
    if (!accessToken || !credentialToDelete) {
      return;
    }
    setIsCredentialDeleting(true);
    try {
      await credentialDeleteCall(accessToken, credentialToDelete.credential_name);
      NotificationsManager.success("Credential deleted successfully");
      await refetchCredentials();
    } catch (error) {
      NotificationsManager.error("Failed to delete credential");
    } finally {
      setCredentialToDelete(null);
      setIsDeleteModalOpen(false);
      setIsCredentialDeleting(false);
    }
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
    <div className="w-full mx-auto flex-auto overflow-y-auto p-2">
      <Button onClick={() => setIsAddModalOpen(true)}>Add Credential</Button>
      <div className="flex justify-between items-center mt-4 mb-4">
        <Text>Configured credentials for different AI providers. Add and manage your API credentials.</Text>
      </div>

      <Card>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Credential Name</TableHeaderCell>
              <TableHeaderCell>Provider</TableHeaderCell>
              <TableHeaderCell>Actions</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {!credentialList || credentialList.length === 0 ? (
              <TableRow>
                <TableCell colSpan={3} className="text-center py-4 text-gray-500">
                  No credentials configured
                </TableCell>
              </TableRow>
            ) : (
              credentialList.map((credential: CredentialItem, index: number) => {
                const githubLogin = credential.credential_info?.github_login;
                return (
                  <TableRow key={index}>
                    <TableCell>
                      {credential.credential_name}
                      {githubLogin && (
                        <span className="ml-2 text-gray-500 text-sm inline-flex items-center gap-1">
                          (
                          <svg
                            viewBox="0 0 16 16"
                            className="w-4 h-4 inline-block"
                            aria-hidden="true"
                            fill="currentColor"
                          >
                            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
                          </svg>
                          <a
                            href={`https://github.com/${githubLogin}`}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {githubLogin}
                          </a>
                          )
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      {renderProviderBadge((credential.credential_info?.custom_llm_provider as string) || "-")}
                    </TableCell>
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
                        onClick={() => openDeleteModal(credential)}
                        className="ml-2"
                      />
                    </TableCell>
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </Card>

      {isAddModalOpen && (
        <AddCredentialsTab
          onAddCredential={handleAddCredential}
          open={isAddModalOpen}
          onCancel={() => {
            setIsAddModalOpen(false);
            refetchCredentials();
          }}
          uploadProps={uploadProps}
        />
      )}
      {isUpdateModalOpen && (
        <EditCredentialsModal
          open={isUpdateModalOpen}
          existingCredential={selectedCredential}
          onUpdateCredential={handleUpdateCredential}
          uploadProps={uploadProps}
          onCancel={() => setIsUpdateModalOpen(false)}
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
};

export default CredentialsPanel;
