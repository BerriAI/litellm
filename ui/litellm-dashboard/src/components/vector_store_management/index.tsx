import React, { useState, useEffect } from "react";
import { Icon, Button as TremorButton, Col, Text, Grid } from "@tremor/react";
import { Form, Input, InputNumber, Modal, Select as AntSelect } from "antd";
import { RefreshIcon } from "@heroicons/react/outline";
import { useRouter, useSearchParams } from "next/navigation";
import {
  vectorStoreListCall,
  vectorStoreDeleteCall,
  credentialListCall,
  CredentialItem,
  qdrantCreateCollectionCall,
  vectorStoreInfoCall,
  detectEmbeddingDimensionCall,
} from "../networking";
import { VectorStore } from "./types";
import VectorStoreTable from "./VectorStoreTable";
import VectorStoreForm from "./VectorStoreForm";
import DeleteResourceModal from "../common_components/DeleteResourceModal";
import VectorStoreInfoView from "./vector_store_info";
import VectorStoreCollectionView from "./VectorStoreCollectionView";
import { isAdminRole } from "@/utils/roles";
import NotificationsManager from "../molecules/notifications_manager";

interface VectorStoreProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

const VectorStoreManagement: React.FC<VectorStoreProps> = ({ accessToken, userID, userRole }) => {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [vectorStores, setVectorStores] = useState<VectorStore[]>([]);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [vectorStoreToDelete, setVectorStoreToDelete] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [credentials, setCredentials] = useState<CredentialItem[]>([]);
  const [selectedVectorStoreId, setSelectedVectorStoreId] = useState<string | null>(null);
  const [editVectorStore, setEditVectorStore] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isCreateCollectionModalVisible, setIsCreateCollectionModalVisible] = useState(false);
  const [collectionVectorStoreId, setCollectionVectorStoreId] = useState<string | null>(null);
  const [collectionConfigJson, setCollectionConfigJson] = useState("{}");
  const [isCreatingCollection, setIsCreatingCollection] = useState(false);
  const [isDetectingVectorSize, setIsDetectingVectorSize] = useState(false);
  const [collectionEmbeddingModel, setCollectionEmbeddingModel] = useState<string | null>(null);
  const [collectionViewVectorStoreId, setCollectionViewVectorStoreId] = useState<string | null>(null);
  const [collectionForm] = Form.useForm();

  const updateQueryParams = (updates: Record<string, string | null>) => {
    const params = new URLSearchParams(searchParams.toString());
    Object.entries(updates).forEach(([key, value]) => {
      if (value === null) {
        params.delete(key);
      } else {
        params.set(key, value);
      }
    });
    if (!params.get("page")) {
      params.set("page", "vector-stores");
    }
    router.replace(`?${params.toString()}`, { scroll: false });
  };

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
    setCollectionViewVectorStoreId(null);
    updateQueryParams({ view: null, vector_store_id: null });
  };

  const handleEdit = (vectorStoreId: string) => {
    setSelectedVectorStoreId(vectorStoreId);
    setEditVectorStore(true);
    setCollectionViewVectorStoreId(null);
    updateQueryParams({ view: null, vector_store_id: null });
  };

  const handleCloseInfo = () => {
    setSelectedVectorStoreId(null);
    setEditVectorStore(false);
    updateQueryParams({ view: null, vector_store_id: null });
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

  const handleCreateCollectionOpen = async (vectorStoreId: string) => {
    setCollectionVectorStoreId(vectorStoreId);
    setCollectionConfigJson("{}");
    setCollectionEmbeddingModel(null);
    collectionForm.setFieldsValue({
      collection_name: vectorStoreId,
      distance: "cosine",
    });
    if (accessToken) {
      try {
        const response = await vectorStoreInfoCall(accessToken, vectorStoreId);
        const embeddingModel =
          response?.vector_store?.litellm_params?.litellm_embedding_model ||
          response?.vector_store?.litellm_params?.embedding_model ||
          null;
        setCollectionEmbeddingModel(embeddingModel);
        if (embeddingModel) {
          setIsDetectingVectorSize(true);
          try {
            const dimension = await detectEmbeddingDimensionCall(accessToken, embeddingModel);
            collectionForm.setFieldsValue({ vector_size: dimension });
          } catch (error) {
            console.error("Error detecting embedding dimension:", error);
            NotificationsManager.fromBackend(`Failed to detect embedding dimension: ${error}`);
          } finally {
            setIsDetectingVectorSize(false);
          }
        }
      } catch (error) {
        console.error("Error fetching vector store info:", error);
      }
    }
    setIsCreateCollectionModalVisible(true);
  };

  const handleCreateCollection = async (values: any) => {
    if (!accessToken || !collectionVectorStoreId) return;
    setIsCreatingCollection(true);
    try {
      let collectionConfig = {};
      try {
        collectionConfig = collectionConfigJson.trim() ? JSON.parse(collectionConfigJson) : {};
      } catch (e) {
        NotificationsManager.fromBackend("Invalid JSON in collection config");
        setIsCreatingCollection(false);
        return;
      }

      const payload = {
        vector_store_id: collectionVectorStoreId,
        collection_name: values.collection_name || collectionVectorStoreId,
        vector_size: values.vector_size,
        distance: values.distance,
        collection_config: collectionConfig,
      };

      await qdrantCreateCollectionCall(accessToken, payload);
      NotificationsManager.success("Qdrant collection created successfully");
      setIsCreateCollectionModalVisible(false);
      setCollectionVectorStoreId(null);
      collectionForm.resetFields();
      fetchVectorStores();
    } catch (error) {
      console.error("Error creating Qdrant collection:", error);
      NotificationsManager.fromBackend("Error creating Qdrant collection: " + error);
    } finally {
      setIsCreatingCollection(false);
    }
  };

  const handleViewCollection = (vectorStoreId: string) => {
    setSelectedVectorStoreId(null);
    setEditVectorStore(false);
    setCollectionViewVectorStoreId(vectorStoreId);
    updateQueryParams({ view: "collection", vector_store_id: vectorStoreId });
  };

  const handleCloseCollectionView = () => {
    setCollectionViewVectorStoreId(null);
    updateQueryParams({ view: null, vector_store_id: null });
  };

  useEffect(() => {
    fetchVectorStores();
    fetchCredentials();
  }, [accessToken]);

  useEffect(() => {
    const viewParam = searchParams.get("view");
    const vectorStoreIdParam = searchParams.get("vector_store_id");
    if (viewParam === "collection" && vectorStoreIdParam) {
      setCollectionViewVectorStoreId(vectorStoreIdParam);
      setSelectedVectorStoreId(null);
      setEditVectorStore(false);
    }
  }, [searchParams]);

  return collectionViewVectorStoreId ? (
    <VectorStoreCollectionView
      vectorStoreId={collectionViewVectorStoreId}
      accessToken={accessToken}
      onClose={handleCloseCollectionView}
    />
  ) : selectedVectorStoreId ? (
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
            <VectorStoreTable
              data={vectorStores}
              onView={handleView}
              onEdit={handleEdit}
              onDelete={handleDelete}
              onCreateCollection={handleCreateCollectionOpen}
              onViewCollection={handleViewCollection}
              showCreateCollection
            />
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

        <Modal
          title="Create Qdrant Collection"
          open={isCreateCollectionModalVisible}
          onCancel={() => setIsCreateCollectionModalVisible(false)}
          onOk={() => collectionForm.submit()}
          confirmLoading={isCreatingCollection}
        >
          <Form form={collectionForm} layout="vertical" onFinish={handleCreateCollection}>
            <Form.Item
              label="Collection Name"
              name="collection_name"
              rules={[{ required: true, message: "Please enter a collection name" }]}
            >
              <Input placeholder="collection-name" />
            </Form.Item>
            <Form.Item
              label="Vector Size"
              name="vector_size"
              rules={[{ required: true, message: "Please enter the vector size" }]}
            >
              <InputNumber
                min={1}
                className="w-full"
                placeholder="4096"
                disabled={isDetectingVectorSize}
              />
            </Form.Item>
            <Form.Item
              label="Distance Metric"
              name="distance"
              rules={[{ required: true, message: "Please select a distance metric" }]}
            >
              <AntSelect
                options={[
                  { value: "cosine", label: "Cosine" },
                  { value: "dot", label: "Dot" },
                  { value: "euclid", label: "Euclid" },
                ]}
              />
            </Form.Item>
            <Form.Item label="Collection Config (JSON)">
              <Input.TextArea
                rows={4}
                value={collectionConfigJson}
                onChange={(e) => setCollectionConfigJson(e.target.value)}
                placeholder='{"hnsw_config": {"m": 16}}'
              />
            </Form.Item>
          </Form>
        </Modal>

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
