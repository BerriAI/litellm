import {
  credentialCreateCall,
  credentialDeleteCall,
  CredentialItem,
  credentialUpdateCall,
} from "@/components/networking";
import { Pencil, Trash2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { UploadProps } from "../add_model/add_model_upload_types";
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

const PROVIDER_BADGE_CLASSES: Record<string, string> = {
  openai: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  azure:
    "bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300",
  anthropic:
    "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  default: "bg-muted text-muted-foreground",
};

const CredentialsPanel: React.FC<CredentialsPanelProps> = ({ uploadProps }) => {
  const { accessToken } = useAuthorized();
  const { data: credentialsResponse, refetch: refetchCredentials } =
    useCredentials();
  const credentialList = credentialsResponse?.credentials || [];

  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isUpdateModalOpen, setIsUpdateModalOpen] = useState(false);
  const [selectedCredential, setSelectedCredential] =
    useState<CredentialItem | null>(null);
  const [credentialToDelete, setCredentialToDelete] =
    useState<CredentialItem | null>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isCredentialDeleting, setIsCredentialDeleting] = useState(false);

  const restrictedFields = ["credential_name", "custom_llm_provider"];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleUpdateCredential = async (values: any) => {
    if (!accessToken) {
      return;
    }

    const filter_credential_values = Object.entries(values)
      .filter(([key]) => !restrictedFields.includes(key))
      .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
    const newCredential = {
      credential_name: values.credential_name,
      credential_values: filter_credential_values,
      credential_info: {
        custom_llm_provider: values.custom_llm_provider,
      },
    };

    await credentialUpdateCall(
      accessToken,
      values.credential_name,
      newCredential,
    );
    NotificationsManager.success("Credential updated successfully");
    setIsUpdateModalOpen(false);
    await refetchCredentials();
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleAddCredential = async (values: any) => {
    if (!accessToken) {
      return;
    }

    const filter_credential_values = Object.entries(values)
      .filter(([key]) => !restrictedFields.includes(key))
      .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
    const newCredential = {
      credential_name: values.credential_name,
      credential_values: filter_credential_values,
      credential_info: {
        custom_llm_provider: values.custom_llm_provider,
      },
    };

    await credentialCreateCall(accessToken, newCredential);
    NotificationsManager.success("Credential added successfully");
    setIsAddModalOpen(false);
    await refetchCredentials();
  };

  const renderProviderBadge = (provider: string) => {
    const classes =
      PROVIDER_BADGE_CLASSES[provider.toLowerCase()] ||
      PROVIDER_BADGE_CLASSES.default;
    return <Badge className={cn("text-xs", classes)}>{provider}</Badge>;
  };

  const handleDeleteCredential = async () => {
    if (!accessToken || !credentialToDelete) {
      return;
    }
    setIsCredentialDeleting(true);
    try {
      await credentialDeleteCall(
        accessToken,
        credentialToDelete.credential_name,
      );
      NotificationsManager.success("Credential deleted successfully");
      await refetchCredentials();
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_error) {
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
        <p className="text-muted-foreground">
          Configured credentials for different AI providers. Add and manage your
          API credentials.
        </p>
      </div>

      <Card>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Credential Name</TableHead>
              <TableHead>Provider</TableHead>
              <TableHead>Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {!credentialList || credentialList.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={3}
                  className="text-center py-4 text-muted-foreground"
                >
                  No credentials configured
                </TableCell>
              </TableRow>
            ) : (
              credentialList.map(
                (credential: CredentialItem, index: number) => (
                  <TableRow key={index}>
                    <TableCell>{credential.credential_name}</TableCell>
                    <TableCell>
                      {renderProviderBadge(
                        (credential.credential_info
                          ?.custom_llm_provider as string) || "-",
                      )}
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => {
                          setSelectedCredential(credential);
                          setIsUpdateModalOpen(true);
                        }}
                        aria-label="Edit credential"
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7 ml-2 text-destructive hover:text-destructive"
                        onClick={() => openDeleteModal(credential)}
                        aria-label="Delete credential"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </TableCell>
                  </TableRow>
                ),
              )
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
        title="Delete Credential?"
        message="Are you sure you want to delete this credential? This action cannot be undone and may break existing integrations."
        resourceInformationTitle="Credential Information"
        resourceInformation={[
          {
            label: "Credential Name",
            value: credentialToDelete?.credential_name,
          },
          {
            label: "Provider",
            value:
              credentialToDelete?.credential_info?.custom_llm_provider || "-",
          },
        ]}
        confirmLoading={isCredentialDeleting}
        requiredConfirmation={credentialToDelete?.credential_name}
      />
    </div>
  );
};

export default CredentialsPanel;
