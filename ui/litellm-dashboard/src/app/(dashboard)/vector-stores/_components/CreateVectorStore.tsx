import React, { useId, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "@/lib/toast";
import { CircleCheck, CircleHelp, Inbox, X } from "lucide-react";
import { v4 as uuidv4 } from "uuid";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/shared/Alert";
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
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import S3VectorsConfig from "./S3VectorsConfig";

const ACCEPTED_DOCUMENT_EXTENSIONS = ".pdf,.txt,.docx,.md,.doc";

const ACCEPTED_DOCUMENT_TYPES = [
  "application/pdf",
  "text/plain",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/msword",
  "text/markdown",
];

const MAX_DOCUMENT_BYTES = 50 * 1024 * 1024;

const RAG_INGEST_UNSUPPORTED_PROVIDERS = new Set(["valkey", "opensearch"]);

const providerItems = Object.entries(VectorStoreProviders)
  .filter(([providerEnum]) => !RAG_INGEST_UNSUPPORTED_PROVIDERS.has(vectorStoreProviderMap[providerEnum]))
  .map(([providerEnum, providerDisplayName]) => ({
    value: vectorStoreProviderMap[providerEnum],
    label: providerDisplayName,
  }));

const asText = (value: unknown): string => (typeof value === "string" ? value : "");

const IngestSuccessAlert: React.FC<{ ingestResults: RAGIngestResponse[] }> = ({ ingestResults }) => {
  const { t } = useTranslation();
  const [dismissed, setDismissed] = useState(false);

  if (dismissed) {
    return null;
  }

  return (
    <Alert variant="success">
      <CircleCheck />
      <AlertTitle>{t("vectorStores.create.success.title")}</AlertTitle>
      <AlertDescription>
        <div>
          <p>
            <strong>{t("vectorStores.create.success.vectorStoreId")}</strong> {ingestResults[0]?.vector_store_id}
          </p>
          <p>
            <strong>{t("vectorStores.create.success.documentsIngested")}</strong> {ingestResults.length}
          </p>
        </div>
      </AlertDescription>
      <AlertAction>
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label={t("vectorStores.create.success.close")}
          onClick={() => setDismissed(true)}
        >
          <X className="size-4" />
        </Button>
      </AlertAction>
    </Alert>
  );
};

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
  const { t } = useTranslation();
  const [documents, setDocuments] = useState<DocumentUpload[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [selectedProvider, setSelectedProvider] = useState<string>("bedrock");
  const [vectorStoreName, setVectorStoreName] = useState<string>("");
  const [vectorStoreDescription, setVectorStoreDescription] = useState<string>("");
  const [ingestResults, setIngestResults] = useState<RAGIngestResponse[]>([]);
  const [providerParams, setProviderParams] = useState<Record<string, unknown>>({});
  const documentsInputId = useId();

  const isSupportedDocument = (file: File): boolean => {
    if (!ACCEPTED_DOCUMENT_TYPES.includes(file.type)) {
      toast.error(t("vectorStores.create.toasts.unsupportedType", { name: file.name }));
      return false;
    }
    if (file.size >= MAX_DOCUMENT_BYTES) {
      toast.error(t("vectorStores.create.toasts.tooLarge", { name: file.name }));
      return false;
    }
    return true;
  };

  const handleAddDocuments = (files: readonly File[]) => {
    const accepted: DocumentUpload[] = files.filter(isSupportedDocument).map((file) => ({
      uid: uuidv4(),
      name: file.name,
      status: "done",
      size: file.size,
      type: file.type,
      originFileObj: file,
    }));

    if (accepted.length > 0) {
      setDocuments((prev) => [...prev, ...accepted]);
    }
  };

  const handleRemoveDocument = (uid: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.uid !== uid));
  };

  const handleCreateVectorStore = async () => {
    if (documents.length === 0) {
      toast.warning(t("vectorStores.create.toasts.noDocuments"));
      return;
    }

    if (!selectedProvider) {
      toast.warning(t("vectorStores.create.toasts.noProvider"));
      return;
    }

    // Validate provider-specific required fields
    const requiredFields = getProviderSpecificFields(selectedProvider).filter((field) => field.required);
    for (const field of requiredFields) {
      if (!providerParams[field.name]) {
        toast.warning(t("vectorStores.create.toasts.fieldRequired", { field: t(field.label) }));
        return;
      }
    }

    // S3 Vectors specific validation
    if (selectedProvider === "s3_vectors") {
      const bucketName = asText(providerParams.vector_bucket_name);
      const indexName = asText(providerParams.index_name);
      if (bucketName && bucketName.length < 3) {
        toast.warning(t("vectorStores.create.toasts.bucketTooShort"));
        return;
      }
      if (indexName && indexName.length > 0 && indexName.length < 3) {
        toast.warning(t("vectorStores.create.toasts.indexTooShort"));
        return;
      }
    }

    if (!accessToken) {
      toast.error(t("vectorStores.create.toasts.noAccessToken"));
      return;
    }

    setIsCreating(true);
    const results: RAGIngestResponse[] = [];
    let vectorStoreId: string | undefined;
    let failure: { name: string; error: unknown } | undefined;

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
          failure = { name: doc.name, error };
          break;
        }
      }

      if (failure && !vectorStoreId) {
        throw failure.error;
      }

      setIngestResults(results);
      if (failure) {
        toast.warning(
          t("vectorStores.create.toasts.partialFailure", {
            id: String(vectorStoreId),
            count: results.length,
            name: failure.name,
          }),
          { description: failure.error instanceof Error ? failure.error.message : String(failure.error) },
        );
      } else {
        toast.success(t("vectorStores.create.toasts.created", { count: results.length, id: String(vectorStoreId) }));
      }

      if (onSuccess && vectorStoreId) {
        onSuccess(vectorStoreId);
      }

      if (failure) {
        return;
      }

      // Clear documents after successful creation
      setTimeout(() => {
        setDocuments([]);
        setIngestResults([]);
      }, 3000);
    } catch (error) {
      console.error("Error creating vector store:", error);
      toast.fromError(t("vectorStores.create.toasts.createFailed", { error: String(error) }));
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <TooltipProvider>
      <div className="space-y-6">
        <div>
          <h3 className="text-lg font-medium">{t("vectorStores.create.title")}</h3>
          <p className="text-sm text-muted-foreground">{t("vectorStores.create.description")}</p>
        </div>

        {/* Upload Area */}
        <Card>
          <CardContent>
            <div className="mb-4">
              <p className="font-medium">{t("vectorStores.create.upload.title")}</p>
              <p className="text-sm text-muted-foreground block mt-1">{t("vectorStores.create.upload.description")}</p>
            </div>
            <label
              htmlFor={documentsInputId}
              className="flex cursor-pointer flex-col items-center gap-2 rounded-md border border-dashed border-input bg-muted/30 px-6 py-10 text-center transition-colors hover:border-primary hover:bg-muted/50 focus-within:border-ring focus-within:ring-3 focus-within:ring-ring/50"
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => {
                event.preventDefault();
                handleAddDocuments(Array.from(event.dataTransfer.files));
              }}
            >
              <Inbox className="size-12 text-primary" />
              <span className="text-base">{t("vectorStores.create.upload.dropzone")}</span>
              <span className="text-sm text-muted-foreground">{t("vectorStores.create.upload.formats")}</span>
              <input
                id={documentsInputId}
                type="file"
                multiple
                accept={ACCEPTED_DOCUMENT_EXTENSIONS}
                className="sr-only"
                onChange={(event) => {
                  handleAddDocuments(Array.from(event.target.files ?? []));
                  event.target.value = "";
                }}
              />
            </label>
          </CardContent>
        </Card>

        {/* Documents Table */}
        {documents.length > 0 && (
          <Card>
            <CardContent>
              <div className="mb-4">
                <p className="font-medium">{t("vectorStores.create.uploadedDocuments", { count: documents.length })}</p>
              </div>
              <DocumentsTable documents={documents} onRemove={handleRemoveDocument} />
            </CardContent>
          </Card>
        )}

        {/* Provider Selection and Vector Store Details */}
        <Card>
          <CardContent className="space-y-4">
            <div>
              <p className="font-medium">{t("vectorStores.create.configure.title")}</p>
              <p className="text-sm text-muted-foreground block mt-1">
                {t("vectorStores.create.configure.description")}
              </p>
            </div>

            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="vector-store-name">
                  {labelWithHint(t("vectorStores.create.fields.name"), t("vectorStores.create.fields.nameHint"))}
                </FieldLabel>
                <Input
                  id="vector-store-name"
                  value={vectorStoreName}
                  onChange={(e) => setVectorStoreName(e.target.value)}
                  placeholder={t("vectorStores.create.fields.namePlaceholder")}
                />
              </Field>

              <Field>
                <FieldLabel htmlFor="vector-store-description">
                  {labelWithHint(
                    t("vectorStores.create.fields.description"),
                    t("vectorStores.create.fields.descriptionHint"),
                  )}
                </FieldLabel>
                <Textarea
                  id="vector-store-description"
                  value={vectorStoreDescription}
                  onChange={(e) => setVectorStoreDescription(e.target.value)}
                  placeholder={t("vectorStores.create.fields.descriptionPlaceholder")}
                  rows={2}
                />
              </Field>

              <Field>
                <FieldLabel htmlFor="vector-store-provider">
                  {labelWithHint(
                    t("vectorStores.create.fields.provider"),
                    t("vectorStores.create.fields.providerHint"),
                  )}
                </FieldLabel>
                <Select
                  items={providerItems}
                  value={selectedProvider}
                  onValueChange={(value: string | null) => value !== null && setSelectedProvider(value)}
                >
                  <SelectTrigger id="vector-store-provider" className="w-full">
                    <SelectValue placeholder={t("vectorStores.create.fields.providerPlaceholder")} />
                  </SelectTrigger>
                  <SelectContent>
                    {providerItems.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        <Logo src={vectorStoreProviderLogoMap[item.label]} label={item.label} className="w-5 h-5" />
                        <span>{item.label}</span>
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
                      {labelWithHint(t(field.label), t(field.tooltip))}
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
                {isCreating ? t("vectorStores.create.buttons.creating") : t("vectorStores.create.buttons.create")}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Success Message */}
        {ingestResults.length > 0 && <IngestSuccessAlert ingestResults={ingestResults} />}
      </div>
    </TooltipProvider>
  );
};

export default CreateVectorStore;
