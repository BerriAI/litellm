import React, { useState, useEffect } from "react";
import {
  Card,
  Icon,
  Button,
  Col,
  Text,
  Grid,
  TextInput,
} from "@tremor/react";
import {
  InformationCircleIcon,
  RefreshIcon,
} from "@heroicons/react/outline";
import {
  Modal,
  Form,
  Select as Select2,
  message,
  Tooltip,
  Input
} from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import VectorStoreInfoView from "./vector_store_info";
import { vectorStoreCreateCall, vectorStoreListCall, vectorStoreDeleteCall } from "../networking";
import { VectorStore } from "./types";
import VectorStoreTable from "./VectorStoreTable";
import { Providers } from "../provider_info_helpers";
interface VectorStoreProps {
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
}

const VectorStoreManagement: React.FC<VectorStoreProps> = ({
  accessToken,
  userID,
  userRole,
}) => {
  const [vectorStores, setVectorStores] = useState<VectorStore[]>([]);
  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false);
  const [selectedVectorStoreId, setSelectedVectorStoreId] = useState<string | null>(null);
  const [editVectorStore, setEditVectorStore] = useState<boolean>(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [vectorStoreToDelete, setVectorStoreToDelete] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [form] = Form.useForm();
  const [metadataJson, setMetadataJson] = useState("{}");

  const fetchVectorStores = async () => {
    if (!accessToken) return;
    try {
      const response = await vectorStoreListCall(accessToken);
      console.log("List vector stores response:", response);
      setVectorStores(response.data || []);
    } catch (error) {
      console.error("Error fetching vector stores:", error);
      message.error("Error fetching vector stores: " + error);
    }
  };

  const handleRefreshClick = () => {
    fetchVectorStores();
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  };

  const handleCreate = async (formValues: any) => {
    if (!accessToken) return;
    try {
      // Parse metadata JSON
      let metadata = {};
      try {
        metadata = metadataJson.trim() ? JSON.parse(metadataJson) : {};
      } catch (e) {
        message.error("Invalid JSON in metadata field");
        return;
      }

      await vectorStoreCreateCall(accessToken, {
        vector_store_id: formValues.vector_store_id,
        custom_llm_provider: formValues.custom_llm_provider,
        vector_store_name: formValues.vector_store_name,
        vector_store_description: formValues.vector_store_description,
        vector_store_metadata: metadata,
      });
      message.success("Vector store created successfully");
      setIsCreateModalVisible(false);
      form.resetFields();
      setMetadataJson("{}");
      fetchVectorStores();
    } catch (error) {
      console.error("Error creating vector store:", error);
      message.error("Error creating vector store: " + error);
    }
  };

  const handleDelete = async (vectorStoreId: string) => {
    setVectorStoreToDelete(vectorStoreId);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (!accessToken || !vectorStoreToDelete) return;
    try {
      await vectorStoreDeleteCall(accessToken, vectorStoreToDelete);
      message.success("Vector store deleted successfully");
      fetchVectorStores();
    } catch (error) {
      console.error("Error deleting vector store:", error);
      message.error("Error deleting vector store: " + error);
    }
    setIsDeleteModalOpen(false);
    setVectorStoreToDelete(null);
  };

  useEffect(() => {
    fetchVectorStores();
  }, [accessToken]);

  return (
    <div className="w-full mx-4 h-[75vh]">
      {selectedVectorStoreId ? (
        <VectorStoreInfoView
          vectorStoreId={selectedVectorStoreId}
          onClose={() => {
            setSelectedVectorStoreId(null);
            setEditVectorStore(false);
          }}
          accessToken={accessToken}
          is_admin={userRole === "Admin"}
          editVectorStore={editVectorStore}
        />
      ) : (
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
            Click on a vector store ID to view and edit its details.

            <p>You can use vector stores to store and retrieve LLM embeddings. Currently, we support Amazon Bedrock vector stores.</p>
          </Text>

          <Button
            className="mb-4"
            onClick={() => setIsCreateModalVisible(true)}
          >
            + Create Vector Store
          </Button>

          <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
            <Col numColSpan={1}>
              <VectorStoreTable
                data={vectorStores}
                onEdit={(vectorStore) => {
                  setSelectedVectorStoreId(vectorStore.vector_store_id);
                  setEditVectorStore(true);
                }}
                onDelete={handleDelete}
                onSelectVectorStore={setSelectedVectorStoreId}
              />
            </Col>
          </Grid>

          {/* Create Vector Store Modal */}
          <Modal
            title="Create New Vector Store"
            visible={isCreateModalVisible}
            width={800}
            footer={null}
            onCancel={() => {
              setIsCreateModalVisible(false);
              form.resetFields();
              setMetadataJson("{}");
            }}
          >
            <Form
              form={form}
              onFinish={handleCreate}
              labelCol={{ span: 8 }}
              wrapperCol={{ span: 16 }}
              labelAlign="left"
            >
              <Form.Item
                label="Vector Store ID"
                name="vector_store_id"
                rules={[{ required: true, message: "Please input a vector store ID" }]}
              >
                <TextInput />
              </Form.Item>

              <Form.Item
                label="Vector Store Name"
                name="vector_store_name"
              >
                <TextInput />
              </Form.Item>

              <Form.Item
                label="Description"
                name="vector_store_description"
              >
                <Input.TextArea rows={4} />
              </Form.Item>

              <Form.Item
                label={
                  <span>
                    Provider{' '}
                    <Tooltip title="Select the provider for this vector store">
                      <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                    </Tooltip>
                  </span>
                }
                name="custom_llm_provider"
                rules={[{ required: true, message: "Please select a provider" }]}
                initialValue="bedrock"
              >
                <Select2>
                  <Select2.Option value="bedrock">
                    {Providers.Bedrock}
                  </Select2.Option>
                </Select2>
              </Form.Item>

              <Form.Item
                label={
                  <span>
                    Metadata{' '}
                    <Tooltip title="JSON metadata for the vector store (optional)">
                      <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                    </Tooltip>
                  </span>
                }
              >
              </Form.Item>

              <Form.Item wrapperCol={{ offset: 8, span: 16 }}>
                <Button type="submit">Create Vector Store</Button>
              </Form.Item>
            </Form>
          </Modal>

          {/* Delete Confirmation Modal */}
          <Modal
            title="Confirm Delete"
            visible={isDeleteModalOpen}
            onOk={confirmDelete}
            onCancel={() => {
              setIsDeleteModalOpen(false);
              setVectorStoreToDelete(null);
            }}
          >
            <p>Are you sure you want to delete the vector store "{vectorStoreToDelete}"?</p>
            <p>This action cannot be undone.</p>
          </Modal>
        </div>
      )}
    </div>
  );
};

export default VectorStoreManagement;
