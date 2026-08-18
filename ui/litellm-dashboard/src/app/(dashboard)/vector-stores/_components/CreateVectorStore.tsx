import React, { useState } from "react";
import { Card, Title, Text } from "@tremor/react";
import { Upload, Alert } from "antd";
import { toast } from "@/lib/toast";
import { InboxOutlined } from "@ant-design/icons";
import type { UploadProps } from "antd";
import { CircleHelp } from "lucide-react";
import { ragIngestCall } from "@/components/networking";
import { DocumentUpload, RAGIngestResponse } from "@/components/vector_store_management/types";
import DocumentsTable from "./DocumentsTable";
import {
  VectorStoreProviders,
  vectorStoreProviderLogoMap,
  vectorStoreProviderMap,
  getProviderSpecificFields,
  VectorStoreFieldConfig,
} from "@/components/vector_store_providers";
import { Logo } from "@/components/molecules/logo/Logo";
import { Field, FieldGroup, FieldLabel } from "@/components/shared/form/field";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import S3VectorsConfig from "./S3VectorsConfig";

const { Dragger } = Upload;

const asText = (value: unknown): string => (typeof value === "string" ? value : "");

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

interface CreateVectorStoreProps {
  accessToken: string | null;
  onSuccess?: (vectorStoreId: string) => void;
}

const CreateVectorStore: React.FC<CreateVectorStoreProps> = ({ accessToken, onSuccess }) => {
  const [documents, setDocuments] = useState<DocumentUpload[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>("bedrock");
  const [vectorStoreName, setVectorStoreName] = useState<string>("");
  const [vectorStoreDescription, setVectorStoreDescription] = useState<string>("");
  const [ingestResults, setIngestResults] = useState<RAGIngestResponse[]>([]);
  const [providerParams, setProviderParams] = useState<Record<string, unknown>>({});

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
        toast.error(`${file.name} is not a supported file type. Please upload PDF, TXT, DOCX, or MD files.`);
        return Upload.LIST_IGNORE;
      }

      const isLt50M = file.size / 1024 / 1024 < 50;
      if (!isLt50M) {
        toast.error(`${file.name} must be smaller than 50MB!`);
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
      toast.warning("Please upload at least one document");
      return;
    }

    if (!selectedProvider) {
      toast.warning("Please select a provider");
      return;
    }

    // Validate provider-specific required fields
    const requiredFields = getProviderSpecificFields(selectedProvider).filter((field) => field.required);
    for (const field of requiredFields) {
      if (!providerParams[field.name]) {
        toast.warning(`Please provide ${field.label}`);
        return;
      }
    }

    // S3 Vectors specific validation
    if (selectedProvider === "s3_vectors") {
      const bucketName = asText(providerParams.vector_bucket_name);
      const indexName = asText(providerParams.index_name);
      if (bucketName && bucketName.length < 3) {
        toast.warning("Vector bucket name must be at least 3 characters");
        return;
      }
      if (indexName && indexName.length > 0 && indexName.length < 3) {
        toast.warning("Index name must be at least 3 characters if provided");
        return;
      }
    }

    if (!accessToken) {
      toast.error("No access token available");
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
      toast.success(
        `Successfully created vector store with ${results.length} document(s). Vector Store ID: ${vectorStoreId}`,
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
      toast.fromError(`Failed to create vector store: ${error}`);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <TooltipProvider>
      <div className="space-y-6">
        <div>
          <Title>Create Vector Store</Title>
          <Text className="text-muted-foreground">
            Upload documents and select a provider to create a new vector store with embedded content.
          </Text>
        </div>

        {/* Upload Area */}
        <Card>
          <div className="mb-4">
            <Text className="font-medium">Step 1: Upload Documents</Text>
            <Text className="text-sm text-muted-foreground block mt-1">
              Upload one or more documents (PDF, TXT, DOCX, MD). Maximum file size: 50MB per file.
            </Text>
          </div>
          <Dragger {...uploadProps}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined style={{ fontSize: "48px", color: "#1890ff" }} />
            </p>
            <p className="ant-upload-text">Click or drag files to this area to upload</p>
            <p className="ant-upload-hint">Support for single or bulk upload. Supported formats: PDF, TXT, DOCX, MD</p>
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
              <Text className="text-sm text-muted-foreground block mt-1">
                Choose the provider and optionally provide a name and description for your vector store.
              </Text>
            </div>

            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="vector-store-name">
                  {labelWithHint("Vector Store Name", "Optional: Give your vector store a meaningful name")}
                </FieldLabel>
                <Input
                  id="vector-store-name"
                  value={vectorStoreName}
                  onChange={(e) => setVectorStoreName(e.target.value)}
                  placeholder="e.g., Product Documentation, Customer Support KB"
                />
              </Field>

              <Field>
                <FieldLabel htmlFor="vector-store-description">
                  {labelWithHint("Description", "Optional: Describe what this vector store contains")}
                </FieldLabel>
                <Textarea
                  id="vector-store-description"
                  value={vectorStoreDescription}
                  onChange={(e) => setVectorStoreDescription(e.target.value)}
                  placeholder="e.g., Contains all product documentation and user guides"
                  rows={2}
                />
              </Field>

              <Field>
                <FieldLabel htmlFor="vector-store-provider">
                  {labelWithHint("Provider", "Select the provider for embedding and vector store operations")}
                </FieldLabel>
                <Select
                  value={selectedProvider}
                  onValueChange={(value: string | null) => value !== null && setSelectedProvider(value)}
                >
                  <SelectTrigger id="vector-store-provider" className="w-full">
                    <SelectValue placeholder="Select a provider" />
                  </SelectTrigger>
                  <SelectContent alignItemWithTrigger={false}>
                    {Object.entries(VectorStoreProviders).map(([providerEnum, providerDisplayName]) => (
                      <SelectItem key={providerEnum} value={vectorStoreProviderMap[providerEnum]}>
                        <Logo
                          src={vectorStoreProviderLogoMap[providerDisplayName]}
                          label={providerDisplayName}
                          className="w-5 h-5"
                        />
                        <span>{providerDisplayName}</span>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>

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
                getProviderSpecificFields(selectedProvider).map((field: VectorStoreFieldConfig) => (
                  <Field key={field.name}>
                    <FieldLabel htmlFor={`vector-store-${field.name}`}>
                      {labelWithHint(field.label, field.tooltip)}
                    </FieldLabel>
                    <Input
                      id={`vector-store-${field.name}`}
                      type={field.type === "password" ? "password" : "text"}
                      value={asText(providerParams[field.name])}
                      onChange={(e) => setProviderParams((prev) => ({ ...prev, [field.name]: e.target.value }))}
                      placeholder={field.placeholder}
                    />
                  </Field>
                ))}
            </FieldGroup>

            <div className="flex justify-end">
              <Button
                size="lg"
                onClick={handleCreateVectorStore}
                disabled={isCreating || documents.length === 0 || !selectedProvider}
              >
                {isCreating && <UiLoadingSpinner className="size-4" />}
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
    </TooltipProvider>
  );
};

export default CreateVectorStore;
