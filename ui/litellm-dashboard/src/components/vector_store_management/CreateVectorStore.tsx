import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Card, Title, Text } from "@tremor/react";
import { Upload, Button, Select, Form, Alert, Tooltip, Input } from "antd";
import MessageManager from "@/components/molecules/message_manager";
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
  const { t } = useTranslation();
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
        MessageManager.error(t("vectorStoreManagement.createVectorStore.unsupportedFileType", { name: file.name }));
        return Upload.LIST_IGNORE;
      }

      const isLt50M = file.size / 1024 / 1024 < 50;
      if (!isLt50M) {
        MessageManager.error(t("vectorStoreManagement.createVectorStore.fileTooLarge", { name: file.name }));
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
      MessageManager.warning(t("vectorStoreManagement.createVectorStore.uploadAtLeastOne"));
      return;
    }

    if (!selectedProvider) {
      MessageManager.warning(t("vectorStoreManagement.createVectorStore.selectProvider"));
      return;
    }

    // Validate provider-specific required fields
    const requiredFields = getProviderSpecificFields(selectedProvider, t).filter((field) => field.required);
    for (const field of requiredFields) {
      if (!providerParams[field.name]) {
        MessageManager.warning(t("vectorStoreManagement.createVectorStore.provideField", { label: field.label }));
        return;
      }
    }

    // S3 Vectors specific validation
    if (selectedProvider === "s3_vectors") {
      if (providerParams.vector_bucket_name && providerParams.vector_bucket_name.length < 3) {
        MessageManager.warning(t("vectorStoreManagement.createVectorStore.bucketNameMinLength"));
        return;
      }
      if (providerParams.index_name && providerParams.index_name.length > 0 && providerParams.index_name.length < 3) {
        MessageManager.warning(t("vectorStoreManagement.createVectorStore.indexNameMinLength"));
        return;
      }
    }

    if (!accessToken) {
      MessageManager.error(t("common.error"));
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
        setDocuments((prev) => prev.map((d) => (d.uid === doc.uid ? { ...d, status: "uploading" as const } : d)));

        try {
          const result = await ragIngestCall(
            accessToken,
            doc.originFileObj,
            selectedProvider,
            vectorStoreId, // Use the same vector store ID for subsequent uploads
            vectorStoreName || undefined,
            vectorStoreDescription || undefined,
            providerParams,
          );

          // Store the vector store ID from the first successful ingest
          if (!vectorStoreId && result.vector_store_id) {
            vectorStoreId = result.vector_store_id;
          }

          results.push(result);

          // Update document status to done
          setDocuments((prev) => prev.map((d) => (d.uid === doc.uid ? { ...d, status: "done" as const } : d)));
        } catch (error) {
          console.error(`Error ingesting ${doc.name}:`, error);
          // Update document status to error
          setDocuments((prev) => prev.map((d) => (d.uid === doc.uid ? { ...d, status: "error" as const } : d)));
          throw error; // Stop processing on first error
        }
      }

      setIngestResults(results);
      NotificationsManager.success(
        t("vectorStoreManagement.createVectorStore.createSuccess", { count: results.length, vectorStoreId }),
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
      NotificationsManager.fromBackend(t("vectorStoreManagement.createVectorStore.createFailed", { error }));
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <Title>{t("vectorStoreManagement.createVectorStore.pageTitle")}</Title>
        <Text className="text-gray-500">{t("vectorStoreManagement.createVectorStore.pageSubtitle")}</Text>
      </div>

      {/* Upload Area */}
      <Card>
        <div className="mb-4">
          <Text className="font-medium">{t("vectorStoreManagement.createVectorStore.step1Title")}</Text>
          <Text className="text-sm text-gray-500 block mt-1">
            {t("vectorStoreManagement.createVectorStore.step1Subtitle")}
          </Text>
        </div>
        <Dragger {...uploadProps}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined style={{ fontSize: "48px", color: "#1890ff" }} />
          </p>
          <p className="ant-upload-text">{t("vectorStoreManagement.createVectorStore.draggerText")}</p>
          <p className="ant-upload-hint">{t("vectorStoreManagement.createVectorStore.draggerHint")}</p>
        </Dragger>
      </Card>

      {/* Documents Table */}
      {documents.length > 0 && (
        <Card>
          <div className="mb-4">
            <Text className="font-medium">
              {t("vectorStoreManagement.createVectorStore.uploadedDocuments", { count: documents.length })}
            </Text>
          </div>
          <DocumentsTable documents={documents} onRemove={handleRemoveDocument} />
        </Card>
      )}

      {/* Provider Selection and Vector Store Details */}
      <Card>
        <div className="space-y-4">
          <div>
            <Text className="font-medium">{t("vectorStoreManagement.createVectorStore.step2Title")}</Text>
            <Text className="text-sm text-gray-500 block mt-1">
              {t("vectorStoreManagement.createVectorStore.step2Subtitle")}
            </Text>
          </div>

          <Form form={form} layout="vertical">
            <Form.Item
              label={
                <span>
                  {t("vectorStoreManagement.createVectorStore.vectorStoreNameLabel")}{" "}
                  <Tooltip title={t("vectorStoreManagement.createVectorStore.vectorStoreNameTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
            >
              <Input
                value={vectorStoreName}
                onChange={(e) => setVectorStoreName(e.target.value)}
                placeholder={t("vectorStoreManagement.createVectorStore.vectorStoreNamePlaceholder")}
                size="large"
                className="rounded-md"
              />
            </Form.Item>

            <Form.Item
              label={
                <span>
                  {t("common.description")}{" "}
                  <Tooltip title={t("vectorStoreManagement.createVectorStore.descriptionTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
            >
              <Input.TextArea
                value={vectorStoreDescription}
                onChange={(e) => setVectorStoreDescription(e.target.value)}
                placeholder={t("vectorStoreManagement.createVectorStore.descriptionPlaceholder")}
                rows={2}
                size="large"
                className="rounded-md"
              />
            </Form.Item>

            <Form.Item
              label={
                <span>
                  {t("vectorStoreManagement.createVectorStore.providerLabel")}{" "}
                  <Tooltip title={t("vectorStoreManagement.createVectorStore.providerTooltip")}>
                    <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                  </Tooltip>
                </span>
              }
              required
            >
              <Select
                value={selectedProvider}
                onChange={setSelectedProvider}
                placeholder={t("common.select")}
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
              getProviderSpecificFields(selectedProvider, t).map((field: VectorStoreFieldConfig) => {
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
                        onChange={(e) => setProviderParams((prev) => ({ ...prev, [field.name]: e.target.value }))}
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
                      onChange={(e) => setProviderParams((prev) => ({ ...prev, [field.name]: e.target.value }))}
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
              {isCreating
                ? t("vectorStoreManagement.createVectorStore.creating")
                : t("vectorStoreManagement.createVectorStore.createButton")}
            </Button>
          </div>
        </div>
      </Card>

      {/* Success Message */}
      {ingestResults.length > 0 && (
        <Alert
          message={t("vectorStoreManagement.createVectorStore.successMessage")}
          description={
            <div>
              <p>
                <strong>{t("vectorStoreManagement.createVectorStore.vectorStoreIdLabel")}</strong>{" "}
                {ingestResults[0]?.vector_store_id}
              </p>
              <p>
                <strong>{t("vectorStoreManagement.createVectorStore.documentsIngestedLabel")}</strong>{" "}
                {ingestResults.length}
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
