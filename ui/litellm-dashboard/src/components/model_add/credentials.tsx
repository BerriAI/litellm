import {
  credentialCreateCall,
  credentialDeleteCall,
  CredentialItem,
  credentialUpdateCall,
  type ProviderCreateInfo,
} from "@/components/networking"; // Assume this is your networking function
import { GithubOutlined } from "@ant-design/icons";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { KeyIcon } from "@heroicons/react/outline";
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
import { Form, Modal } from "antd";
import { UploadProps } from "antd/es/upload";
import { useCallback, useState } from "react";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import NotificationsManager from "../molecules/notifications_manager";
import AddCredentialsTab from "./AddCredentialModal";
import EditCredentialsModal from "./EditCredentialModal";
import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useDeviceCodeFlow } from "@/hooks/useDeviceCodeFlow";

const DEVICE_CODE_PROVIDERS: Record<string, string> = {
  github_copilot: "GitHub Copilot",
  chatgpt: "ChatGPT",
};
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

  // --- Re-authorize flow for device-code providers ---
  const [reauthorizingCredential, setReauthorizingCredential] = useState<CredentialItem | null>(null);

  const reauthorizeProviderInfo: ProviderCreateInfo | null = (() => {
    const provider = reauthorizingCredential?.credential_info?.custom_llm_provider;
    if (!provider || !(provider in DEVICE_CODE_PROVIDERS)) return null;
    return {
      provider,
      provider_display_name: DEVICE_CODE_PROVIDERS[provider],
      litellm_provider: provider,
      credential_fields: [],
      auth_flow: "device_code",
    };
  })();

  const handleReauthorizeSuccess = useCallback(
    async (apiKey: string) => {
      if (!accessToken || !reauthorizingCredential) throw new Error("Missing context");
      await credentialUpdateCall(accessToken, reauthorizingCredential.credential_name, {
        credential_name: reauthorizingCredential.credential_name,
        credential_values: { api_key: apiKey },
        credential_info: reauthorizingCredential.credential_info,
      });
      NotificationsManager.success("Credential re-authorized successfully");
      await refetchCredentials();
    },
    [accessToken, reauthorizingCredential, refetchCredentials],
  );

  const {
    state: reauthorizeState,
    reset: resetReauthorize,
    renderUI: renderReauthorizeFlow,
  } = useDeviceCodeFlow({
    accessToken,
    providerInfo: reauthorizeProviderInfo,
    onSuccess: handleReauthorizeSuccess,
  });

  const closeReauthorizeModal = () => {
    resetReauthorize();
    setReauthorizingCredential(null);
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
                          (<GithubOutlined className="w-4 h-4" />
                          <a
                            href={`https://github.com/${githubLogin}`}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            {githubLogin}
                          </a>)
                        </span>
                      )}
                    </TableCell>
                    <TableCell>
                      {renderProviderBadge((credential.credential_info?.custom_llm_provider as string) || "-")}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        {credential.credential_info?.custom_llm_provider &&
                          (credential.credential_info.custom_llm_provider in DEVICE_CODE_PROVIDERS) && (
                          <Button
                            icon={KeyIcon}
                            variant="light"
                            size="sm"
                            tooltip="Re-authorize"
                            onClick={() => setReauthorizingCredential(credential)}
                          />
                        )}
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
                        />
                      </div>
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

      <Modal
        title={`Re-authorize ${reauthorizingCredential?.credential_name ?? ""}`}
        open={reauthorizingCredential != null}
        onCancel={closeReauthorizeModal}
        footer={reauthorizeState.phase === "success" ? (
          <Button onClick={closeReauthorizeModal}>Done</Button>
        ) : null}
        width={500}
      >
        {renderReauthorizeFlow()}
      </Modal>

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
