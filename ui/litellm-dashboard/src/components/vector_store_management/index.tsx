import React, { useState, useEffect } from "react";
import { Icon, Button as TremorButton, Col, Text, Grid } from "@tremor/react";
import { RefreshIcon } from "@heroicons/react/outline";
import { vectorStoreListCall, vectorStoreDeleteCall, credentialListCall, CredentialItem } from "../networking";
import { VectorStore } from "./types";
import VectorStoreTable from "./VectorStoreTable";
import VectorStoreForm from "./VectorStoreForm";
import DeleteModal from "./DeleteModal";
import VectorStoreInfoView from "./vector_store_info";
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

  const fetchVectorStores = async () => {
    if (!accessToken) return;
    try {
      const response = await vectorStoreListCall(accessToken);
      console.log("List vector stores response:", response);
      setVectorStores(response.data || []);
    } catch (error) {
      console.error("Error fetching vector stores:", error);
      NotificationsManager.fromBackend("Error fetching vector stores: " + error);
    }
  };

  const fetchCredentials = async () => {
    if (!accessToken) return;
    try {
      const response = await credentialListCall(accessToken);
      console.log("List credentials response:", response);
      setCredentials(response.credentials || []);
    } catch (error) {
      console.error("Error fetching credentials:", error);
      NotificationsManager.fromBackend("Error fetching credentials: " + error);
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
    try {
      await vectorStoreDeleteCall(accessToken, vectorStoreToDelete);
      NotificationsManager.success("Vector store deleted successfully");
      fetchVectorStores();
    } catch (error) {
      console.error("Error deleting vector store:", error);
      NotificationsManager.fromBackend("Error deleting vector store: " + error);
    }
    setIsDeleteModalOpen(false);
    setVectorStoreToDelete(null);
  };

  const handleCreateSuccess = () => {
    setIsCreateModalVisible(false);
    fetchVectorStores();
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
    <div className="w-full mx-4 h-[75vh]">
      <div className="gap-2 p-8 h-[75vh] w-full mt-2">
        <div className="flex justify-between mt-2 w-full items-center mb-4">
          <h1>Vector Store Management</h1>
          <div className="flex items-center space-x-2">
            {lastRefreshed && <Text>Last Refreshed: {lastRefreshed}</Text>}
            <Icon
              icon={RefreshIcon}
              variant="shadow"
              size="xs"
              className="self-center cursor-pointer"
              onClick={handleRefreshClick}
            />
          </div>
        </div>

        <Text className="mb-4">
          <p>You can use vector stores to store and retrieve LLM embeddings..</p>
        </Text>

        <TremorButton className="mb-4" onClick={() => setIsCreateModalVisible(true)}>
          + Add Vector Store
        </TremorButton>

        <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
          <Col numColSpan={1}>
            <VectorStoreTable data={vectorStores} onView={handleView} onEdit={handleEdit} onDelete={handleDelete} />
          </Col>
        </Grid>

        {/* Create Vector Store Modal */}
        <VectorStoreForm
          isVisible={isCreateModalVisible}
          onCancel={() => setIsCreateModalVisible(false)}
          onSuccess={handleCreateSuccess}
          accessToken={accessToken}
          credentials={credentials}
        />

        {/* Delete Confirmation Modal */}
        <DeleteModal
          isVisible={isDeleteModalOpen}
          onCancel={() => setIsDeleteModalOpen(false)}
          onConfirm={confirmDelete}
        />
      </div>
    </div>
  );
};

export default VectorStoreManagement;
