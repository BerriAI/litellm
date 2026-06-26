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
import { useTranslation } from "react-i18next";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import NotificationsManager from "../molecules/notifications_manager";
import AddCredentialsTab from "./AddCredentialModal";
import EditCredentialsModal from "./EditCredentialModal";
import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
interface CredentialsPanelProps {
  uploadProps: UploadProps;
}

const CredentialsPanel: React.FC<CredentialsPanelProps> = ({ uploadProps }) => {
  const { t } = useTranslation();
  const { accessToken, userRole } = useAuthorized();
  // Admin Viewer follows the read-parity rule: see credentials, do not modify.
  const canModifyCredentials = isProxyAdminRole(userRole ?? "");
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
    NotificationsManager.success(t("modelAdd.credentials.updatedSuccess"));
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
    NotificationsManager.success(t("modelAdd.credentials.addedSuccess"));
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
      NotificationsManager.success(t("modelAdd.credentials.deletedSuccess"));
      await refetchCredentials();
    } catch (error) {
      NotificationsManager.error(t("modelAdd.credentials.deleteFailed"));
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
      {canModifyCredentials && (
        <Button onClick={() => setIsAddModalOpen(true)}>{t("modelAdd.credentials.addCredential")}</Button>
      )}
      <div className="flex justify-between items-center mt-4 mb-4">
        <Text>{t("modelAdd.credentials.configuredCredentials")}</Text>
      </div>

      <Card>
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>{t("modelAdd.credentials.credentialName")}</TableHeaderCell>
              <TableHeaderCell>{t("modelAdd.credentials.provider")}</TableHeaderCell>
              <TableHeaderCell>{t("common.actions")}</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {!credentialList || credentialList.length === 0 ? (
              <TableRow>
                <TableCell colSpan={4} className="text-center py-4 text-gray-500">
                  {t("modelAdd.credentials.noCredentials")}
                </TableCell>
              </TableRow>
            ) : (
              credentialList.map((credential: CredentialItem, index: number) => (
                <TableRow key={index}>
                  <TableCell>{credential.credential_name}</TableCell>
                  <TableCell>
                    {renderProviderBadge((credential.credential_info?.custom_llm_provider as string) || "-")}
                  </TableCell>
                  <TableCell>
                    {canModifyCredentials ? (
                      <>
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
                      </>
                    ) : null}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </Card>

      {isAddModalOpen && (
        <AddCredentialsTab
          onAddCredential={handleAddCredential}
          open={isAddModalOpen}
          onCancel={() => setIsAddModalOpen(false)}
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
        title={t("modelAdd.credentials.deleteTitle")}
        message={t("modelAdd.credentials.deleteMessage")}
        resourceInformationTitle={t("modelAdd.credentials.credentialInformation")}
        resourceInformation={[
          { label: t("modelAdd.credentials.credentialName"), value: credentialToDelete?.credential_name },
          {
            label: t("modelAdd.credentials.provider"),
            value: credentialToDelete?.credential_info?.custom_llm_provider || "-",
          },
        ]}
        confirmLoading={isCredentialDeleting}
        requiredConfirmation={credentialToDelete?.credential_name}
      />
    </div>
  );
};

export default CredentialsPanel;
