import React, { useState } from "react";
import { Card, Title, Text } from "@tremor/react";
import { Upload, Button, Select, Form, message, Alert } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import type { UploadProps } from "antd";
import { ragIngestCall } from "../networking";
import { DocumentUpload, RAGIngestResponse } from "./types";
import DocumentsTable from "./DocumentsTable";
import { Providers, provider_map } from "../provider_info_helpers";
import { ProviderLogo } from "../molecules/models/ProviderLogo";
import NotificationsManager from "../molecules/notifications_manager";

const { Dragger } = Upload;

interface CreateVectorStoreProps {
  accessToken: string | null;
  onSuccess?: (vectorStoreId: string) => void;
}

const CreateVectorStore: React.FC<CreateVectorStoreProps> = ({ accessToken, onSuccess }) => {
  const [form] = Form.useForm();
  const [documents, setDocuments] = useState<DocumentUpload[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>("openai");
  const [ingestResults, setIngestResults] = useState<RAGIngestResponse[]>([]);

  const uploadProps: UploadProps = {
    name: "file",
    multiple: true,
    accept: ".pdf,.txt,.docx,.md,.doc",
    beforeUpload: (file) => {
      const isValidType = [
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "text/markdown",
      ].includes(file.type);

      if (!isValidType) {
        message.error(`${file.name} is not a supported file type. Please upload PDF, TXT, DOCX, or MD files.`);
        return Upload.LIST_IGNORE;
      }

      const isLt50M = file.size / 1024 / 1024 < 50;
      if (!isLt50M) {
        message.error(`${file.name} must be smaller than 50MB!`);
        return Upload.LIST_IGNORE;
      }

      const newDoc: DocumentUpload = {
        uid: file.uid,
        name: file.name,
        status: "done",
        size: file.size,
        type: file.type,
        originFileObj: file,
      };

      setDocuments((prev) => [...prev, newDoc]);
      return false; // Prevent auto upload
    },
    onRemove: (file) => {
      setDocuments((prev) => prev.filter((doc) => doc.uid !== file.uid));
    },
    fileList: documents.map((doc) => ({
      uid: doc.uid,
      name: doc.name,
      status: doc.status,
      size: doc.size,
    })),
    showUploadList: false, // We'll use our custom table
  };

  const handleRemoveDocument = (uid: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.uid !== uid));
  };

  const handleCreateVectorStore = async () => {
    if (documents.length === 0) {
      message.warning("Please upload at least one document");
      return;
    }

    if (!selectedProvider) {
      message.warning("Please select a provider");
      return;
    }

    if (!accessToken) {
      message.error("No access token available");
      return;
    }

    setIsCreating(true);
    const results: RAGIngestResponse[] = [];
    let vectorStoreId: string | undefined;

    try {
      // Ingest each document
      for (const doc of documents) {
        if (!doc.originFileObj) continue;

        // Update document status to uploading
        setDocuments((prev) =>
          prev.map((d) => (d.uid === doc.uid ? { ...d, status: "uploading" as const } : d))
        );

        try {
          const result = await ragIngestCall(
            accessToken,
            doc.originFileObj,
            selectedProvider,
            vectorStoreId // Use the same vector store ID for subsequent uploads
          );

          // Store the vector store ID from the first successful ingest
          if (!vectorStoreId && result.vector_store_id) {
            vectorStoreId = result.vector_store_id;
          }

          results.push(result);

          // Update document status to done
          setDocuments((prev) =>
            prev.map((d) => (d.uid === doc.uid ? { ...d, status: "done" as const } : d))
          );
        } catch (error) {
          console.error(`Error ingesting ${doc.name}:`, error);
          // Update document status to error
          setDocuments((prev) =>
            prev.map((d) => (d.uid === doc.uid ? { ...d, status: "error" as const } : d))
          );
          throw error; // Stop processing on first error
        }
      }

      setIngestResults(results);
      NotificationsManager.success(
        `Successfully created vector store with ${results.length} document(s). Vector Store ID: ${vectorStoreId}`
      );

      if (onSuccess && vectorStoreId) {
        onSuccess(vectorStoreId);
      }

      // Clear documents after successful creation
      setTimeout(() => {
        setDocuments([]);
        setIngestResults([]);
      }, 3000);
    } catch (error) {
      console.error("Error creating vector store:", error);
      NotificationsManager.fromBackend(`Failed to create vector store: ${error}`);
    } finally {
      setIsCreating(false);
    }
  };

  const supportedProviders = ["openai", "bedrock"]; // Add more as needed

  return (
    <div className="space-y-6">
      <div>
        <Title>Create Vector Store</Title>
        <Text className="text-gray-500">
          Upload documents and select a provider to create a new vector store with embedded content.
        </Text>
      </div>

      {/* Upload Area */}
      <Card>
        <div className="mb-4">
          <Text className="font-medium">Step 1: Upload Documents</Text>
          <Text className="text-sm text-gray-500 block mt-1">
            Upload one or more documents (PDF, TXT, DOCX, MD). Maximum file size: 50MB per file.
          </Text>
        </div>
        <Dragger {...uploadProps}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: "48px", color: "#1890ff" }} />
          </p>
          <p className="ant-upload-text">Click or drag files to this area to upload</p>
          <p className="ant-upload-hint">
            Support for single or bulk upload. Supported formats: PDF, TXT, DOCX, MD
          </p>
        </Dragger>
      </Card>

      {/* Documents Table */}
      {documents.length > 0 && (
        <Card>
          <div className="mb-4">
            <Text className="font-medium">Uploaded Documents ({documents.length})</Text>
          </div>
          <DocumentsTable documents={documents} onRemove={handleRemoveDocument} />
        </Card>
      )}

      {/* Provider Selection and Create Button */}
      <Card>
        <div className="space-y-4">
          <div>
            <Text className="font-medium">Step 2: Select Provider</Text>
            <Text className="text-sm text-gray-500 block mt-1">
              Choose the LLM provider for embedding and vector store operations.
            </Text>
          </div>

          <Form form={form} layout="vertical">
            <Form.Item label="Provider" required>
              <Select
                value={selectedProvider}
                onChange={setSelectedProvider}
                placeholder="Select a provider"
                size="large"
                style={{ width: "100%" }}
              >
                {Object.entries(Providers)
                  .filter(([providerEnum]) => supportedProviders.includes(provider_map[providerEnum]?.toLowerCase()))
                  .map(([providerEnum, providerDisplayName]) => {
                    const providerValue = provider_map[providerEnum]?.toLowerCase();
                    return (
                      <Select.Option key={providerEnum} value={providerValue}>
                        <div className="flex items-center space-x-2">
                          <ProviderLogo provider={providerEnum} className="w-5 h-5" />
                          <span>{providerDisplayName}</span>
                        </div>
                      </Select.Option>
                    );
                  })}
              </Select>
            </Form.Item>
          </Form>

          <div className="flex justify-end">
            <Button
              type="primary"
              size="large"
              onClick={handleCreateVectorStore}
              loading={isCreating}
              disabled={documents.length === 0 || !selectedProvider}
            >
              {isCreating ? "Creating Vector Store..." : "Create Vector Store"}
            </Button>
          </div>
        </div>
      </Card>

      {/* Success Message */}
      {ingestResults.length > 0 && (
        <Alert
          message="Vector Store Created Successfully"
          description={
            <div>
              <p>
                <strong>Vector Store ID:</strong> {ingestResults[0]?.vector_store_id}
              </p>
              <p>
                <strong>Documents Ingested:</strong> {ingestResults.length}
              </p>
            </div>
          }
          type="success"
          showIcon
          closable
        />
      )}
    </div>
  );
};

export default CreateVectorStore;
