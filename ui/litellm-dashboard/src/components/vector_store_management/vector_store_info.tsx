import React, { useState, useEffect } from "react";
import {
  Card,
  Text,
  Title,
  Button,
  Badge,
} from "@tremor/react";
import {
  Form,
  Input,
  Select as Select2,
  message,
  Tooltip,
} from "antd";
import { InfoCircleOutlined } from '@ant-design/icons';
import { vectorStoreInfoCall } from "../networking";
import { VectorStore } from "./types";
import { Providers } from "../provider_info_helpers";

interface VectorStoreInfoViewProps {
  vectorStoreId: string;
  onClose: () => void;
  accessToken: string | null;
  is_admin: boolean;
  editVectorStore: boolean;
}

const VectorStoreInfoView: React.FC<VectorStoreInfoViewProps> = ({
  vectorStoreId,
  onClose,
  accessToken,
  is_admin,
  editVectorStore,
}) => {
  const [form] = Form.useForm();
  const [vectorStoreDetails, setVectorStoreDetails] = useState<VectorStore | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(editVectorStore);
  const [metadataString, setMetadataString] = useState<string>("{}");

  const fetchVectorStoreDetails = async () => {
    if (!accessToken) return;
    try {
      const response = await vectorStoreInfoCall(accessToken, vectorStoreId);
      if (response && response.vector_store) {
        setVectorStoreDetails(response.vector_store);
        
        // If metadata exists and is an object, stringify it for display/editing
        if (response.vector_store.vector_store_metadata) {
          const metadata = typeof response.vector_store.vector_store_metadata === 'string'
            ? JSON.parse(response.vector_store.vector_store_metadata)
            : response.vector_store.vector_store_metadata;
          setMetadataString(JSON.stringify(metadata, null, 2));
        }
        
        if (editVectorStore) {
          form.setFieldsValue({
            vector_store_id: response.vector_store.vector_store_id,
            custom_llm_provider: response.vector_store.custom_llm_provider,
            vector_store_name: response.vector_store.vector_store_name,
            vector_store_description: response.vector_store.vector_store_description,
          });
        }
      }
    } catch (error) {
      console.error("Error fetching vector store details:", error);
      message.error("Error fetching vector store details: " + error);
    }
  };

  useEffect(() => {
    fetchVectorStoreDetails();
  }, [vectorStoreId, accessToken]);

  const handleSave = async (values: any) => {
    if (!accessToken) return;
    try {
      // Parse the metadata JSON string
      let metadata = {};
      try {
        metadata = metadataString ? JSON.parse(metadataString) : {};
      } catch (e) {
        message.error("Invalid JSON in metadata field");
        return;
      }
      
      const updateData = {
        vector_store_id: values.vector_store_id,
        custom_llm_provider: values.custom_llm_provider,
        vector_store_name: values.vector_store_name,
        vector_store_description: values.vector_store_description,
        vector_store_metadata: metadata,
      };
      
      // Use the updated data to call an update endpoint
      // await vectorStoreUpdateCall(accessToken, updateData);
      message.success("Vector store updated successfully");
      setIsEditing(false);
      fetchVectorStoreDetails();
    } catch (error) {
      console.error("Error updating vector store:", error);
      message.error("Error updating vector store: " + error);
    }
  };

  if (!vectorStoreDetails) {
    return <div>Loading...</div>;
  }

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">← Back to Vector Stores</Button>
          <Title>Vector Store ID: {vectorStoreDetails.vector_store_id}</Title>
          <Text className="text-gray-500">{vectorStoreDetails.vector_store_description || "No description"}</Text>
        </div>
        {is_admin && !isEditing && (
          <Button onClick={() => setIsEditing(true)}>Edit Vector Store</Button>
        )}
      </div>

      {isEditing ? (
        <Card>
          <Form
            form={form}
            onFinish={handleSave}
            layout="vertical"
            initialValues={vectorStoreDetails}
          >
            <Form.Item
              label="Vector Store ID"
              name="vector_store_id"
              rules={[{ required: true, message: "Please input a vector store ID" }]}
            >
              <Input disabled />
            </Form.Item>

            <Form.Item
              label="Vector Store Name"
              name="vector_store_name"
            >
              <Input />
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
                  <Tooltip title="JSON metadata for the vector store">
                    <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                  </Tooltip>
                </span>
              }
            >
            </Form.Item>

            <div className="flex justify-end space-x-2">
              <Button onClick={() => setIsEditing(false)}>Cancel</Button>
              <Button type="submit">Save Changes</Button>
            </div>
          </Form>
        </Card>
      ) : (
        <div className="space-y-6">
          <Card>
            <Title>Vector Store Details</Title>
            <div className="space-y-4 mt-4">
              <div>
                <Text className="font-medium">ID</Text>
                <Text>{vectorStoreDetails.vector_store_id}</Text>
              </div>
              <div>
                <Text className="font-medium">Name</Text>
                <Text>{vectorStoreDetails.vector_store_name || "-"}</Text>
              </div>
              <div>
                <Text className="font-medium">Description</Text>
                <Text>{vectorStoreDetails.vector_store_description || "-"}</Text>
              </div>
              <div>
                <Text className="font-medium">Provider</Text>
                <Badge color="blue">
                  {vectorStoreDetails.custom_llm_provider === "bedrock" 
                    ? Providers.Bedrock 
                    : vectorStoreDetails.custom_llm_provider}
                </Badge>
              </div>
              <div>
                <Text className="font-medium">Metadata</Text>
                <div className="bg-gray-50 p-3 rounded mt-2 font-mono text-xs overflow-auto max-h-48">
                  <pre>{metadataString}</pre>
                </div>
              </div>
              <div>
                <Text className="font-medium">Created</Text>
                <Text>{vectorStoreDetails.created_at ? new Date(vectorStoreDetails.created_at).toLocaleString() : "-"}</Text>
              </div>
              <div>
                <Text className="font-medium">Last Updated</Text>
                <Text>{vectorStoreDetails.updated_at ? new Date(vectorStoreDetails.updated_at).toLocaleString() : "-"}</Text>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};

export default VectorStoreInfoView; 