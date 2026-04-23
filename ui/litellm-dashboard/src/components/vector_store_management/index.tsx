import React, { useCallback, useState, useEffect } from "react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { RefreshCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  vectorStoreListCall,
  vectorStoreDeleteCall,
  credentialListCall,
  CredentialItem,
} from "../networking";
import { VectorStore } from "./types";
import VectorStoreTable from "./VectorStoreTable";
import VectorStoreForm from "./VectorStoreForm";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import VectorStoreInfoView from "./vector_store_info";
import CreateVectorStore from "./CreateVectorStore";
import TestVectorStoreTab from "./TestVectorStoreTab";
import { isAdminRole } from "@/utils/roles";
import NotificationsManager from "../molecules/notifications_manager";

interface VectorStoreProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

const VectorStoreManagement: React.FC<VectorStoreProps> = ({ accessToken, userID, userRole }) => {
  const [vectorStores, setVectorStores] = useState<VectorStore[]>([]);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [vectorStoreToDelete, setVectorStoreToDelete] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [credentials, setCredentials] = useState<CredentialItem[]>([]);
  const [selectedVectorStoreId, setSelectedVectorStoreId] = useState<string | null>(null);
  const [editVectorStore, setEditVectorStore] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const fetchVectorStores = useCallback(async () => {
    if (!accessToken) return;
    try {
      const response = await vectorStoreListCall(accessToken);
      setVectorStores(response.data || []);
    } catch (error) {
      console.error("Error fetching vector stores:", error);
      NotificationsManager.fromBackend(
        "Error fetching vector stores: " + error,
      );
    }
  }, [accessToken]);

  const fetchCredentials = useCallback(async () => {
    if (!accessToken) return;
    try {
      const response = await credentialListCall(accessToken);
      setCredentials(response.credentials || []);
    } catch (error) {
      console.error("Error fetching credentials:", error);
      NotificationsManager.fromBackend("Error fetching credentials: " + error);
    }
  }, [accessToken]);

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
      NotificationsManager.success("Vector store deleted successfully");
      fetchVectorStores();
    } catch (error) {
      console.error("Error deleting vector store:", error);
      NotificationsManager.fromBackend("Error deleting vector store: " + error);
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
    console.log("Vector store created:", vectorStoreId);
    fetchVectorStores();
    // Optionally switch to the manage tab
  };

  useEffect(() => {
    fetchVectorStores();
    fetchCredentials();
  }, [fetchVectorStores, fetchCredentials]);

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
    <div className="w-full mx-4 h-[75vh]">
      <div className="gap-2 p-8 h-[75vh] w-full mt-2">
        <div className="flex justify-between mt-2 w-full items-center mb-4">
          <h1>Vector Store Management</h1>
          <div className="flex items-center space-x-2">
            {lastRefreshed && (
              <span className="text-sm text-muted-foreground">
                Last Refreshed: {lastRefreshed}
              </span>
            )}
            <button
              type="button"
              onClick={handleRefreshClick}
              className="p-1.5 rounded-md border border-border shadow-sm hover:bg-muted transition-colors self-center"
              aria-label="Refresh"
            >
              <RefreshCcw className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        <p className="mb-4 text-sm">
          You can use vector stores to store and retrieve LLM embeddings.
        </p>

        <TabGroup>
          <TabList className="mb-6">
            <Tab>Create Vector Store</Tab>
            <Tab>Manage Vector Stores</Tab>
            <Tab>Test Vector Store</Tab>
          </TabList>

          <TabPanels>
            {/* Tab 1: Create Vector Store */}
            <TabPanel>
              <CreateVectorStore accessToken={accessToken} onSuccess={handleVectorStoreCreated} />
            </TabPanel>

            {/* Tab 2: Manage Vector Stores */}
            <TabPanel>
              <Button
                className="mb-4"
                onClick={() => setIsCreateModalVisible(true)}
              >
                + Add Vector Store
              </Button>

              <div className="grid grid-cols-1 gap-2 pt-2 pb-2 w-full mt-2">
                <div className="col-span-1">
                  <VectorStoreTable
                    data={vectorStores}
                    onView={handleView}
                    onEdit={handleEdit}
                    onDelete={handleDelete}
                  />
                </div>
              </div>
            </TabPanel>

            {/* Tab 3: Test Vector Store */}
            <TabPanel>
              <TestVectorStoreTab accessToken={accessToken} vectorStores={vectorStores} />
            </TabPanel>
          </TabPanels>
        </TabGroup>

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
          title="Delete Vector Store"
          message="Are you sure you want to delete this vector store? This action cannot be undone."
          resourceInformationTitle="Vector Store Information"
          resourceInformation={[{ label: "Vector Store ID", value: vectorStoreToDelete, code: true }]}
          onCancel={() => setIsDeleteModalOpen(false)}
          onOk={confirmDelete}
          confirmLoading={isDeleting}
        />
      </div>
    </div>
  );
};

export default VectorStoreManagement;
