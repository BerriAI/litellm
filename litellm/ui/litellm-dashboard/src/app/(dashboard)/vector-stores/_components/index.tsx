import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { RefreshCw } from "lucide-react";
import {
  vectorStoreListCall,
  vectorStoreDeleteCall,
  credentialListCall,
  CredentialItem,
} from "@/components/networking";
import { VectorStore } from "@/components/vector_store_management/types";
import VectorStoreTable from "./VectorStoreTable";
import VectorStoreForm from "./VectorStoreForm";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import VectorStoreInfoView from "./vector_store_info";
import CreateVectorStore from "./CreateVectorStore";
import TestVectorStoreTab from "./TestVectorStoreTab";
import IndexesTab from "./IndexesTab";
import { isAdminRole, isProxyAdminRole } from "@/utils/roles";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent } from "@/components/ui/tabs";
import { PageTabsList as TabsList, PageTabsTrigger as TabsTrigger } from "@/components/shared/Page";
import { useVisitedTabs } from "@/hooks/useVisitedTabs";

interface VectorStoreProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  isViewOnly: boolean;
}

const VectorStoreManagement: React.FC<VectorStoreProps> = ({ accessToken, userID, userRole, isViewOnly }) => {
  const { t } = useTranslation();
  const [vectorStores, setVectorStores] = useState<VectorStore[]>([]);
  const [isLoadingVectorStores, setIsLoadingVectorStores] = useState(true);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [vectorStoreToDelete, setVectorStoreToDelete] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [credentials, setCredentials] = useState<CredentialItem[]>([]);
  const [selectedVectorStoreId, setSelectedVectorStoreId] = useState<string | null>(null);
  const [editVectorStore, setEditVectorStore] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const canCreateVectorStores = isProxyAdminRole(userRole || "") && !isViewOnly;
  const defaultTab = canCreateVectorStores ? "create" : "manage";
  const { onTabChange, hasVisited } = useVisitedTabs(defaultTab);

  const fetchVectorStores = async () => {
    if (!accessToken) {
      setIsLoadingVectorStores(false);
      return;
    }
    try {
      const response = await vectorStoreListCall(accessToken);
      setVectorStores(response.data || []);
    } catch (error) {
      console.error("Error fetching vector stores:", error);
      toast.fromError(t("vectorStores.page.toasts.fetchStoresFailed", { error: String(error) }));
    } finally {
      setIsLoadingVectorStores(false);
    }
  };

  const fetchCredentials = async () => {
    if (!accessToken || !canCreateVectorStores) return;
    try {
      const response = await credentialListCall(accessToken);
      setCredentials(response.credentials || []);
    } catch (error) {
      console.error("Error fetching credentials:", error);
      toast.fromError(t("vectorStores.page.toasts.fetchCredentialsFailed", { error: String(error) }));
    }
  };

  const handleRefreshClick = () => {
    fetchVectorStores();
    fetchCredentials();
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  };

  const handleDelete = async (vectorStoreId: string) => {
    setVectorStoreToDelete(vectorStoreId);
    setIsDeleteModalOpen(true);
  };

  const handleView = (vectorStoreId: string) => {
    setSelectedVectorStoreId(vectorStoreId);
    setEditVectorStore(false);
  };

  const handleEdit = (vectorStoreId: string) => {
    setSelectedVectorStoreId(vectorStoreId);
    setEditVectorStore(true);
  };

  const handleCloseInfo = () => {
    setSelectedVectorStoreId(null);
    setEditVectorStore(false);
    fetchVectorStores();
  };

  const confirmDelete = async () => {
    if (!accessToken || !vectorStoreToDelete) return;
    setIsDeleting(true);
    try {
      await vectorStoreDeleteCall(accessToken, vectorStoreToDelete);
      toast.success(t("vectorStores.page.toasts.deleted"));
      fetchVectorStores();
    } catch (error) {
      console.error("Error deleting vector store:", error);
      toast.fromError(t("vectorStores.page.toasts.deleteFailed", { error: String(error) }));
    } finally {
      setIsDeleting(false);
      setIsDeleteModalOpen(false);
      setVectorStoreToDelete(null);
    }
  };

  const handleCreateSuccess = () => {
    setIsCreateModalVisible(false);
    fetchVectorStores();
  };

  const handleVectorStoreCreated = (vectorStoreId: string) => {
    fetchVectorStores();
    // Optionally switch to the manage tab
  };

  useEffect(() => {
    fetchVectorStores();
    fetchCredentials();
  }, [accessToken]);

  return selectedVectorStoreId ? (
    <div className="w-full h-full">
      <VectorStoreInfoView
        vectorStoreId={selectedVectorStoreId}
        onClose={handleCloseInfo}
        accessToken={accessToken}
        is_admin={isAdminRole(userRole || "")}
        editVectorStore={editVectorStore}
      />
    </div>
  ) : (
    <div className="min-w-0 w-full">
      <div className="min-w-0 w-full p-4 sm:p-6">
        <div className="flex flex-wrap justify-between w-full items-start gap-3 mb-1.5">
          <h1 className="text-2xl font-semibold tracking-tight text-foreground">{t("vectorStores.page.title")}</h1>
          <div className="flex items-center space-x-2">
            {lastRefreshed && (
              <p className="text-sm text-muted-foreground">
                {t("vectorStores.page.lastRefreshed", { time: lastRefreshed })}
              </p>
            )}
            <Button
              variant="outline"
              size="icon-sm"
              aria-label={t("vectorStores.page.refresh")}
              onClick={handleRefreshClick}
            >
              <RefreshCw className="size-4" />
            </Button>
          </div>
        </div>

        <p className="mb-4 text-sm text-muted-foreground">{t("vectorStores.page.description")}</p>

        <Tabs defaultValue={defaultTab} onValueChange={onTabChange}>
          <TabsList className="mb-6">
            {canCreateVectorStores && <TabsTrigger value="create">{t("vectorStores.page.tabs.create")}</TabsTrigger>}
            <TabsTrigger value="manage">{t("vectorStores.page.tabs.manage")}</TabsTrigger>
            <TabsTrigger value="test">{t("vectorStores.page.tabs.test")}</TabsTrigger>
            {isProxyAdminRole(userRole || "") && (
              <TabsTrigger value="indexes">{t("vectorStores.page.tabs.indexes")}</TabsTrigger>
            )}
          </TabsList>

          {canCreateVectorStores && (
            <TabsContent keepMounted={hasVisited("create")} value="create">
              <CreateVectorStore accessToken={accessToken} onSuccess={handleVectorStoreCreated} />
            </TabsContent>
          )}

          <TabsContent keepMounted={hasVisited("manage")} value="manage">
            {canCreateVectorStores && (
              <Button className="mb-4" onClick={() => setIsCreateModalVisible(true)}>
                {t("vectorStores.page.addVectorStore")}
              </Button>
            )}

            <div className="grid grid-cols-1 gap-2 pt-2 pb-2 w-full mt-2">
              <VectorStoreTable
                data={vectorStores}
                isLoading={isLoadingVectorStores}
                onView={handleView}
                onEdit={handleEdit}
                onDelete={handleDelete}
              />
            </div>
          </TabsContent>

          <TabsContent keepMounted={hasVisited("test")} value="test">
            <TestVectorStoreTab accessToken={accessToken} vectorStores={vectorStores} />
          </TabsContent>

          {isProxyAdminRole(userRole || "") && (
            <TabsContent keepMounted={hasVisited("indexes")} value="indexes">
              <IndexesTab accessToken={accessToken} vectorStores={vectorStores} onViewVectorStore={handleView} />
            </TabsContent>
          )}
        </Tabs>

        {/* Create Vector Store Modal */}
        <VectorStoreForm
          isVisible={isCreateModalVisible}
          onCancel={() => setIsCreateModalVisible(false)}
          onSuccess={handleCreateSuccess}
          accessToken={accessToken}
          credentials={credentials}
        />

        {/* Delete Confirmation Modal */}
        <DeleteResourceModal
          isOpen={isDeleteModalOpen}
          title={t("vectorStores.page.deleteModal.title")}
          message={t("vectorStores.page.deleteModal.message")}
          resourceInformationTitle={t("vectorStores.page.deleteModal.resourceTitle")}
          resourceInformation={[
            { label: t("vectorStores.page.deleteModal.vectorStoreId"), value: vectorStoreToDelete, code: true },
          ]}
          onCancel={() => setIsDeleteModalOpen(false)}
          onOk={confirmDelete}
          confirmLoading={isDeleting}
        />
      </div>
    </div>
  );
};

export default VectorStoreManagement;
