import React, { useRef, useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import MessageManager from "@/components/molecules/message_manager";
import { cn } from "@/lib/utils";
import { Inbox, Info } from "lucide-react";
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

interface CreateVectorStoreProps {
  accessToken: string | null;
  onSuccess?: (vectorStoreId: string) => void;
}

const ACCEPTED_TYPES = [
  "application/pdf",
  "text/plain",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/msword",
  "text/markdown",
];

function FieldLabel({
  children,
  tooltip,
  htmlFor,
  required,
}: {
  children: React.ReactNode;
  tooltip: string;
  htmlFor?: string;
  required?: boolean;
}) {
  return (
    <Label htmlFor={htmlFor} className="flex items-center gap-1">
      <span>
        {children}
        {required && <span className="text-destructive"> *</span>}
      </span>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Info className="h-3 w-3 text-muted-foreground" />
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">{tooltip}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </Label>
  );
}

const CreateVectorStore: React.FC<CreateVectorStoreProps> = ({ accessToken, onSuccess }) => {
  const [documents, setDocuments] = useState<DocumentUpload[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>("bedrock");
  const [vectorStoreName, setVectorStoreName] = useState<string>("");
  const [vectorStoreDescription, setVectorStoreDescription] = useState<string>("");
  const [ingestResults, setIngestResults] = useState<RAGIngestResponse[]>([]);
  const [providerParams, setProviderParams] = useState<Record<string, any>>({});
  const [isDragActive, setIsDragActive] = useState(false);
  const [showSuccessAlert, setShowSuccessAlert] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const acceptFiles = (files: FileList | File[]) => {
    const accepted: DocumentUpload[] = [];
    for (const file of Array.from(files)) {
      if (!ACCEPTED_TYPES.includes(file.type)) {
        MessageManager.error(
          `${file.name} is not a supported file type. Please upload PDF, TXT, DOCX, or MD files.`,
        );
        continue;
      }
      if (file.size / 1024 / 1024 >= 50) {
        MessageManager.error(`${file.name} must be smaller than 50MB!`);
        continue;
      }
      accepted.push({
        uid: `${file.name}-${file.size}-${file.lastModified}-${Math.random()}`,
        name: file.name,
        status: "done",
        size: file.size,
        type: file.type,
        originFileObj: file,
      });
    }
    if (accepted.length > 0) {
      setDocuments((prev) => [...prev, ...accepted]);
    }
  };

  const handleRemoveDocument = (uid: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.uid !== uid));
  };

  const handleCreateVectorStore = async () => {
    if (documents.length === 0) {
      MessageManager.warning("Please upload at least one document");
      return;
    }

    if (!selectedProvider) {
      MessageManager.warning("Please select a provider");
      return;
    }

    const requiredFields = getProviderSpecificFields(selectedProvider).filter((field) => field.required);
    for (const field of requiredFields) {
      if (!providerParams[field.name]) {
        MessageManager.warning(`Please provide ${field.label}`);
        return;
      }
    }

    if (selectedProvider === "s3_vectors") {
      if (providerParams.vector_bucket_name && providerParams.vector_bucket_name.length < 3) {
        MessageManager.warning("Vector bucket name must be at least 3 characters");
        return;
      }
      if (providerParams.index_name && providerParams.index_name.length > 0 && providerParams.index_name.length < 3) {
        MessageManager.warning("Index name must be at least 3 characters if provided");
        return;
      }
    }

    if (!accessToken) {
      MessageManager.error("No access token available");
      return;
    }

    setIsCreating(true);
    const results: RAGIngestResponse[] = [];
    let vectorStoreId: string | undefined;

    try {
      for (const doc of documents) {
        if (!doc.originFileObj) continue;

        setDocuments((prev) =>
          prev.map((d) => (d.uid === doc.uid ? { ...d, status: "uploading" as const } : d)),
        );

        try {
          const result = await ragIngestCall(
            accessToken,
            doc.originFileObj,
            selectedProvider,
            vectorStoreId,
            vectorStoreName || undefined,
            vectorStoreDescription || undefined,
            providerParams,
          );

          if (!vectorStoreId && result.vector_store_id) {
            vectorStoreId = result.vector_store_id;
          }

          results.push(result);

          setDocuments((prev) =>
            prev.map((d) => (d.uid === doc.uid ? { ...d, status: "done" as const } : d)),
          );
        } catch (error) {
          console.error(`Error ingesting ${doc.name}:`, error);
          setDocuments((prev) =>
            prev.map((d) => (d.uid === doc.uid ? { ...d, status: "error" as const } : d)),
          );
          throw error;
        }
      }

      setIngestResults(results);
      setShowSuccessAlert(true);
      NotificationsManager.success(
        `Successfully created vector store with ${results.length} document(s). Vector Store ID: ${vectorStoreId}`,
      );

      if (onSuccess && vectorStoreId) {
        onSuccess(vectorStoreId);
      }

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

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer?.files?.length) {
      acceptFiles(e.dataTransfer.files);
    }
  };

  const renderProviderField = (field: VectorStoreFieldConfig) => {
    const fieldId = `provider-field-${field.name}`;
    if (field.type === "select") {
      return (
        <div key={field.name} className="mb-4 space-y-1">
          <FieldLabel htmlFor={fieldId} tooltip={field.tooltip} required={field.required}>
            {field.label}
          </FieldLabel>
          <Input
            id={fieldId}
            value={providerParams[field.name] || ""}
            onChange={(e) =>
              setProviderParams((prev) => ({ ...prev, [field.name]: e.target.value }))
            }
            placeholder={field.placeholder}
            className="rounded-md"
          />
        </div>
      );
    }

    return (
      <div key={field.name} className="mb-4 space-y-1">
        <FieldLabel htmlFor={fieldId} tooltip={field.tooltip} required={field.required}>
          {field.label}
        </FieldLabel>
        <Input
          id={fieldId}
          type={field.type === "password" ? "password" : "text"}
          value={providerParams[field.name] || ""}
          onChange={(e) =>
            setProviderParams((prev) => ({ ...prev, [field.name]: e.target.value }))
          }
          placeholder={field.placeholder}
          className="rounded-md"
        />
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-semibold m-0">Create Vector Store</h2>
        <p className="text-muted-foreground text-sm m-0">
          Upload documents and select a provider to create a new vector store with embedded content.
        </p>
      </div>

      <Card className="p-6">
        <div className="mb-4">
          <p className="font-medium">Step 1: Upload Documents</p>
          <p className="text-sm text-muted-foreground mt-1">
            Upload one or more documents (PDF, TXT, DOCX, MD). Maximum file size: 50MB per file.
          </p>
        </div>
        <div
          role="button"
          tabIndex={0}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragActive(true);
          }}
          onDragLeave={() => setIsDragActive(false)}
          onDrop={handleDrop}
          className={cn(
            "flex flex-col items-center justify-center gap-2 border-2 border-dashed rounded-md p-8 cursor-pointer transition-colors",
            isDragActive
              ? "border-primary bg-accent/50"
              : "border-border hover:border-primary",
          )}
        >
          <Inbox className="h-12 w-12 text-primary" />
          <p className="text-sm font-medium">Click or drag files to this area to upload</p>
          <p className="text-xs text-muted-foreground">
            Support for single or bulk upload. Supported formats: PDF, TXT, DOCX, MD
          </p>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.docx,.md,.doc"
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.length) acceptFiles(e.target.files);
              if (fileInputRef.current) fileInputRef.current.value = "";
            }}
          />
        </div>
      </Card>

      {documents.length > 0 && (
        <Card className="p-6">
          <div className="mb-4">
            <p className="font-medium">Uploaded Documents ({documents.length})</p>
          </div>
          <DocumentsTable documents={documents} onRemove={handleRemoveDocument} />
        </Card>
      )}

      <Card className="p-6">
        <div className="space-y-4">
          <div>
            <p className="font-medium">Step 2: Configure Vector Store</p>
            <p className="text-sm text-muted-foreground mt-1">
              Choose the provider and optionally provide a name and description for your vector store.
            </p>
          </div>

          <div className="space-y-4">
            <div className="space-y-1">
              <FieldLabel
                htmlFor="vector-store-name"
                tooltip="Optional: Give your vector store a meaningful name"
              >
                Vector Store Name
              </FieldLabel>
              <Input
                id="vector-store-name"
                value={vectorStoreName}
                onChange={(e) => setVectorStoreName(e.target.value)}
                placeholder="e.g., Product Documentation, Customer Support KB"
                className="rounded-md"
              />
            </div>

            <div className="space-y-1">
              <FieldLabel
                htmlFor="vector-store-description"
                tooltip="Optional: Describe what this vector store contains"
              >
                Description
              </FieldLabel>
              <Textarea
                id="vector-store-description"
                value={vectorStoreDescription}
                onChange={(e) => setVectorStoreDescription(e.target.value)}
                placeholder="e.g., Contains all product documentation and user guides"
                rows={2}
                className="rounded-md"
              />
            </div>

            <div className="space-y-1">
              <FieldLabel
                htmlFor="vector-store-provider"
                tooltip="Select the provider for embedding and vector store operations"
                required
              >
                Provider
              </FieldLabel>
              <Select value={selectedProvider} onValueChange={setSelectedProvider}>
                <SelectTrigger id="vector-store-provider" className="w-full">
                  <SelectValue placeholder="Select a provider" />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(VectorStoreProviders).map(
                    ([providerEnum, providerDisplayName]) => (
                      <SelectItem
                        key={providerEnum}
                        value={vectorStoreProviderMap[providerEnum]}
                      >
                        <div className="flex items-center space-x-2">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={vectorStoreProviderLogoMap[providerDisplayName]}
                            alt={`${providerEnum} logo`}
                            className="w-5 h-5"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              const parent = target.parentElement;
                              if (parent) {
                                const fallbackDiv = document.createElement("div");
                                fallbackDiv.className =
                                  "w-5 h-5 rounded-full bg-muted flex items-center justify-center text-xs";
                                fallbackDiv.textContent = providerDisplayName.charAt(0);
                                parent.replaceChild(fallbackDiv, target);
                              }
                            }}
                          />
                          <span>{providerDisplayName}</span>
                        </div>
                      </SelectItem>
                    ),
                  )}
                </SelectContent>
              </Select>
            </div>

            {selectedProvider === "s3_vectors" && (
              <S3VectorsConfig
                accessToken={accessToken}
                providerParams={providerParams}
                onParamsChange={setProviderParams}
              />
            )}

            {selectedProvider !== "s3_vectors" &&
              getProviderSpecificFields(selectedProvider).map(renderProviderField)}
          </div>

          <div className="flex justify-end">
            <Button
              onClick={handleCreateVectorStore}
              disabled={isCreating || documents.length === 0 || !selectedProvider}
            >
              {isCreating ? "Creating Vector Store..." : "Create Vector Store"}
            </Button>
          </div>
        </div>
      </Card>

      {ingestResults.length > 0 && showSuccessAlert && (
        <Alert>
          <AlertTitle className="flex items-center justify-between">
            <span>Vector Store Created Successfully</span>
            <button
              type="button"
              onClick={() => setShowSuccessAlert(false)}
              className="text-muted-foreground hover:text-foreground"
              aria-label="Dismiss"
            >
              ×
            </button>
          </AlertTitle>
          <AlertDescription>
            <div>
              <p>
                <strong>Vector Store ID:</strong> {ingestResults[0]?.vector_store_id}
              </p>
              <p>
                <strong>Documents Ingested:</strong> {ingestResults.length}
              </p>
            </div>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

export default CreateVectorStore;
