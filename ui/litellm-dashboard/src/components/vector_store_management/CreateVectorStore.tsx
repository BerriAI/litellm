import React, { useState } from "react";
import { Card, Title, Text } from "@tremor/react";
import { Upload, Button, Select, Form, message, Alert, Tooltip, Input } from "antd";
import { InboxOutlined, InfoCircleOutlined } from "@ant-design/icons";
import type { UploadProps } from "antd";
import { ragIngestCall } from "../networking";
import { DocumentUpload, RAGIngestResponse } from "./types";
import DocumentsTable from "./DocumentsTable";
import {
  VectorStoreProviders,
  vectorStoreProviderLogoMap,
  vectorStoreProviderMap,
  getProviderSpecificFields,
  VectorStoreFieldConfig,
} from "../vector_store_providers";
import NotificationsManager from "../molecules/notifications_manager";
import S3VectorsConfig from "./S3VectorsConfig";

const { Dragger } = Upload;

interface CreateVectorStoreProps {
  accessToken: string | null;
  onSuccess?: (vectorStoreId: string) => void;
}

const CreateVectorStore: React.FC<CreateVectorStoreProps> = ({ accessToken, onSuccess }) => {
  const [form] = Form.useForm();
  const [documents, setDocuments] = useState<DocumentUpload[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>("bedrock");
  const [vectorStoreName, setVectorStoreName] = useState<string>("");
  const [vectorStoreDescription, setVectorStoreDescription] = useState<string>("");
  const [ingestResults, setIngestResults] = useState<RAGIngestResponse[]>([]);
  const [providerParams, setProviderParams] = useState<Record<string, any>>({});

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

    // Validate provider-specific required fields
    const requiredFields = getProviderSpecificFields(selectedProvider).filter((field) => field.required);
    for (const field of requiredFields) {
      if (!providerParams[field.name]) {
        message.warning(`Please provide ${field.label}`);
        return;
      }
    }

    // S3 Vectors specific validation
    if (selectedProvider === "s3_vectors") {
      if (providerParams.vector_bucket_name && providerParams.vector_bucket_name.length < 3) {
        message.warning("Vector bucket name must be at least 3 characters");
        return;
      }
      if (providerParams.index_name && providerParams.index_name.length > 0 && providerParams.index_name.length < 3) {
        message.warning("Index name must be at least 3 characters if provided");
        return;
      }
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
            vectorStoreId, // Use the same vector store ID for subsequent uploads
            vectorStoreName || undefined,
            vectorStoreDescription || undefined,
            providerParams
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

      {/* Provider Selection and Vector Store Details */}
      <Card>
        <div className="space-y-4">
          <div>
            <Text className="font-medium">Step 2: Configure Vector Store</Text>
            <Text className="text-sm text-gray-500 block mt-1">
              Choose the provider and optionally provide a name and description for your vector store.
            </Text>
          </div>

          <Form form={form} layout="vertical">
            <Form.Item
              label={
                <span>
                  Vector Store Name{" "}
                  <Tooltip title="Optional: Give your vector store a meaningful name">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
            >
              <Input
                value={vectorStoreName}
                onChange={(e) => setVectorStoreName(e.target.value)}
                placeholder="e.g., Product Documentation, Customer Support KB"
                size="large"
                className="rounded-md"
              />
            </Form.Item>

            <Form.Item
              label={
                <span>
                  Description{" "}
                  <Tooltip title="Optional: Describe what this vector store contains">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
            >
              <Input.TextArea
                value={vectorStoreDescription}
                onChange={(e) => setVectorStoreDescription(e.target.value)}
                placeholder="e.g., Contains all product documentation and user guides"
                rows={2}
                size="large"
                className="rounded-md"
              />
            </Form.Item>

            <Form.Item
              label={
                <span>
                  Provider{" "}
                  <Tooltip title="Select the provider for embedding and vector store operations">
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              required
            >
              <Select
                value={selectedProvider}
                onChange={setSelectedProvider}
                placeholder="Select a provider"
                size="large"
                style={{ width: "100%" }}
              >
                {Object.entries(VectorStoreProviders).map(([providerEnum, providerDisplayName]) => {
                  return (
                    <Select.Option key={providerEnum} value={vectorStoreProviderMap[providerEnum]}>
                      <div className="flex items-center space-x-2">
                        <img
                          src={vectorStoreProviderLogoMap[providerDisplayName]}
                          alt={`${providerEnum} logo`}
                          className="w-5 h-5"
                          onError={(e) => {
                            // Create a div with provider initial as fallback
                            const target = e.target as HTMLImageElement;
                            const parent = target.parentElement;
                            if (parent) {
                              const fallbackDiv = document.createElement("div");
                              fallbackDiv.className =
                                "w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs";
                              fallbackDiv.textContent = providerDisplayName.charAt(0);
                              parent.replaceChild(fallbackDiv, target);
                            }
                          }}
                        />
                        <span>{providerDisplayName}</span>
                      </div>
                    </Select.Option>
                  );
                })}
              </Select>
            </Form.Item>

            {/* S3 Vectors Configuration */}
            {selectedProvider === "s3_vectors" && (
              <S3VectorsConfig
                accessToken={accessToken}
                providerParams={providerParams}
                onParamsChange={setProviderParams}
              />
            )}

            {/* Other Provider-specific fields */}
            {selectedProvider !== "s3_vectors" &&
              getProviderSpecificFields(selectedProvider).map((field: VectorStoreFieldConfig) => {
                if (field.type === "select") {
                  // For embedding model selection, we'd need to fetch available models
                  // For now, provide a text input as fallback
                  return (
                    <Form.Item
                      key={field.name}
                      label={
                        <span>
                          {field.label}{" "}
                          <Tooltip title={field.tooltip}>
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </Tooltip>
                        </span>
                      }
                      required={field.required}
                    >
                      <Input
                        value={providerParams[field.name] || ""}
                        onChange={(e) =>
                          setProviderParams((prev) => ({ ...prev, [field.name]: e.target.value }))
                        }
                        placeholder={field.placeholder}
                        size="large"
                        className="rounded-md"
                      />
                    </Form.Item>
                  );
                }

                return (
                  <Form.Item
                    key={field.name}
                    label={
                      <span>
                        {field.label}{" "}
                        <Tooltip title={field.tooltip}>
                          <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                        </Tooltip>
                      </span>
                    }
                    required={field.required}
                  >
                    <Input
                      type={field.type === "password" ? "password" : "text"}
                      value={providerParams[field.name] || ""}
                      onChange={(e) =>
                        setProviderParams((prev) => ({ ...prev, [field.name]: e.target.value }))
                      }
                      placeholder={field.placeholder}
                      size="large"
                      className="rounded-md"
                    />
                  </Form.Item>
                );
              })}
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
