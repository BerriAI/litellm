import React, { useEffect, useState } from "react";
import { Controller, useForm } from "react-hook-form";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { Info } from "lucide-react";
import { CredentialItem, vectorStoreCreateCall } from "../networking";
import {
  VectorStoreProviders,
  vectorStoreProviderLogoMap,
  vectorStoreProviderMap,
  getProviderSpecificFields,
  VectorStoreFieldConfig,
} from "../vector_store_providers";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";
import NotificationsManager from "../molecules/notifications_manager";

interface VectorStoreFormProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  credentials: CredentialItem[];
}

interface FormValues {
  custom_llm_provider: string;
  vector_store_id: string;
  vector_store_name?: string;
  vector_store_description?: string;
  litellm_credential_name?: string | null;
  [key: string]: any;
}

function LabelWithTooltip({
  children,
  tooltip,
  required,
  htmlFor,
}: {
  children: React.ReactNode;
  tooltip: string;
  required?: boolean;
  htmlFor?: string;
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

const VectorStoreForm: React.FC<VectorStoreFormProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  accessToken,
  credentials,
}) => {
  const {
    register,
    handleSubmit,
    control,
    reset,
    watch,
    setValue,
    formState: { errors },
  } = useForm<FormValues>({
    defaultValues: {
      custom_llm_provider: "bedrock",
      vector_store_id: "",
      vector_store_name: "",
      vector_store_description: "",
      litellm_credential_name: null,
    },
  });

  const [metadataJson, setMetadataJson] = useState("{}");
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);

  const selectedProvider = watch("custom_llm_provider");

  useEffect(() => {
    if (!accessToken) return;

    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        if (uniqueModels.length > 0) {
          setModelInfo(uniqueModels);
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    loadModels();
  }, [accessToken]);

  const resetAndClose = () => {
    reset();
    setMetadataJson("{}");
  };

  const handleCreate = async (formValues: FormValues) => {
    if (!accessToken) return;
    try {
      let metadata: Record<string, any> = {};
      try {
        metadata = metadataJson.trim() ? JSON.parse(metadataJson) : {};
      } catch {
        NotificationsManager.fromBackend("Invalid JSON in metadata field");
        return;
      }

      const payload: Record<string, any> = {
        vector_store_id: formValues.vector_store_id,
        custom_llm_provider: formValues.custom_llm_provider,
        vector_store_name: formValues.vector_store_name,
        vector_store_description: formValues.vector_store_description,
        vector_store_metadata: metadata,
        litellm_credential_name: formValues.litellm_credential_name,
      };

      const providerFields = getProviderSpecificFields(formValues.custom_llm_provider);
      const litellmParams = providerFields.reduce(
        (acc, field) => {
          if (formValues.custom_llm_provider === "milvus" && field.name === "embedding_model") {
            acc["litellm_embedding_model"] = formValues[field.name];
          } else {
            acc[field.name] = formValues[field.name];
          }
          return acc;
        },
        {} as Record<string, any>,
      );

      payload["litellm_params"] = litellmParams;

      await vectorStoreCreateCall(accessToken, payload);
      NotificationsManager.success("Vector store created successfully");
      resetAndClose();
      onSuccess();
    } catch (error) {
      console.error("Error creating vector store:", error);
      NotificationsManager.fromBackend("Error creating vector store: " + error);
    }
  };

  const handleClose = () => {
    resetAndClose();
    onCancel();
  };

  const embeddingModels = modelInfo
    .filter((option) => option.mode === "embedding" || option.mode === null)
    .map((option) => ({
      value: option.model_group,
      label: option.model_group,
    }));

  const renderProviderField = (field: VectorStoreFieldConfig) => {
    const fieldId = `provider-field-${field.name}`;
    if (field.type === "select") {
      return (
        <div key={field.name} className="space-y-1">
          <LabelWithTooltip htmlFor={fieldId} tooltip={field.tooltip} required={field.required}>
            {field.label}
          </LabelWithTooltip>
          <Controller
            control={control}
            name={field.name}
            rules={
              field.required
                ? { required: `Please select the ${field.label.toLowerCase()}` }
                : undefined
            }
            render={({ field: rhf }) => (
              <Select value={rhf.value || undefined} onValueChange={rhf.onChange}>
                <SelectTrigger id={fieldId} className="w-full">
                  <SelectValue placeholder={field.placeholder || `Select ${field.label}`} />
                </SelectTrigger>
                <SelectContent>
                  {embeddingModels.length === 0 ? (
                    <div className="px-3 py-2 text-sm text-muted-foreground">
                      No options available
                    </div>
                  ) : (
                    embeddingModels.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))
                  )}
                </SelectContent>
              </Select>
            )}
          />
          {errors[field.name] && (
            <p className="text-xs text-destructive">
              {String((errors[field.name] as any)?.message)}
            </p>
          )}
        </div>
      );
    }

    return (
      <div key={field.name} className="space-y-1">
        <LabelWithTooltip htmlFor={fieldId} tooltip={field.tooltip} required={field.required}>
          {field.label}
        </LabelWithTooltip>
        <Input
          id={fieldId}
          type={field.type || "text"}
          placeholder={field.placeholder}
          {...register(field.name, {
            required: field.required
              ? `Please input the ${field.label.toLowerCase()}`
              : false,
          })}
        />
        {errors[field.name] && (
          <p className="text-xs text-destructive">
            {String((errors[field.name] as any)?.message)}
          </p>
        )}
      </div>
    );
  };

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(open) => {
        if (!open) handleClose();
      }}
    >
      <DialogContent className="sm:max-w-[1000px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add New Vector Store</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit(handleCreate)} className="space-y-4">
          <div className="space-y-1">
            <LabelWithTooltip
              htmlFor="vs-provider"
              tooltip="Select the provider for this vector store"
              required
            >
              Provider
            </LabelWithTooltip>
            <Controller
              control={control}
              name="custom_llm_provider"
              rules={{ required: "Please select a provider" }}
              render={({ field }) => (
                <Select value={field.value} onValueChange={field.onChange}>
                  <SelectTrigger id="vs-provider" className="w-full">
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
              )}
            />
            {errors.custom_llm_provider && (
              <p className="text-xs text-destructive">
                {errors.custom_llm_provider.message as string}
              </p>
            )}
          </div>

          {selectedProvider === "pg_vector" && (
            <Alert>
              <AlertTitle>PG Vector Setup Required</AlertTitle>
              <AlertDescription>
                <p>LiteLLM provides a server to connect to PG Vector. To use this provider:</p>
                <ol className="ml-4 mt-2 list-decimal">
                  <li>
                    Deploy the litellm-pgvector server from:{" "}
                    <a
                      href="https://github.com/BerriAI/litellm-pgvector"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline"
                    >
                      https://github.com/BerriAI/litellm-pgvector
                    </a>
                  </li>
                  <li>Configure your PostgreSQL database with pgvector extension</li>
                  <li>Start the server and note the API base URL and API key</li>
                  <li>Enter those details in the fields below</li>
                </ol>
              </AlertDescription>
            </Alert>
          )}

          {selectedProvider === "vertex_rag_engine" && (
            <Alert>
              <AlertTitle>Vertex AI RAG Engine Setup</AlertTitle>
              <AlertDescription>
                <p>To use Vertex AI RAG Engine:</p>
                <ol className="ml-4 mt-2 list-decimal">
                  <li>
                    Set up your Vertex AI RAG Engine corpus following the guide:{" "}
                    <a
                      href="https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="underline"
                    >
                      Vertex AI RAG Engine Overview
                    </a>
                  </li>
                  <li>Create a corpus in your Google Cloud project</li>
                  <li>Note the corpus ID from the Vertex AI console</li>
                  <li>Enter the corpus ID in the Vector Store ID field below</li>
                </ol>
              </AlertDescription>
            </Alert>
          )}

          <div className="space-y-1">
            <LabelWithTooltip
              htmlFor="vs-id"
              tooltip="Enter the vector store ID from your api provider"
              required
            >
              Vector Store ID
            </LabelWithTooltip>
            <Input
              id="vs-id"
              placeholder={
                selectedProvider === "vertex_rag_engine"
                  ? "6917529027641081856 (Get corpus ID from Vertex AI console)"
                  : "Enter vector store ID from your provider"
              }
              {...register("vector_store_id", {
                required: "Please input the vector store ID from your api provider",
              })}
            />
            {errors.vector_store_id && (
              <p className="text-xs text-destructive">
                {errors.vector_store_id.message as string}
              </p>
            )}
          </div>

          {getProviderSpecificFields(selectedProvider).map(renderProviderField)}

          <div className="space-y-1">
            <LabelWithTooltip
              htmlFor="vs-name"
              tooltip="Custom name you want to give to the vector store, this name will be rendered on the LiteLLM UI"
            >
              Vector Store Name
            </LabelWithTooltip>
            <Input id="vs-name" {...register("vector_store_name")} />
          </div>

          <div className="space-y-1">
            <Label htmlFor="vs-description">Description</Label>
            <Textarea id="vs-description" rows={4} {...register("vector_store_description")} />
          </div>

          <div className="space-y-1">
            <LabelWithTooltip
              htmlFor="vs-credential"
              tooltip="Optionally select API provider credentials for this vector store eg. Bedrock API KEY"
            >
              Existing Credentials
            </LabelWithTooltip>
            <Controller
              control={control}
              name="litellm_credential_name"
              render={({ field }) => (
                <Select
                  value={field.value ?? "__none__"}
                  onValueChange={(v) => {
                    field.onChange(v === "__none__" ? null : v);
                    setValue("litellm_credential_name", v === "__none__" ? null : v);
                  }}
                >
                  <SelectTrigger id="vs-credential" className="w-full">
                    <SelectValue placeholder="Select or search for existing credentials" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">None</SelectItem>
                    {credentials.map((credential) => (
                      <SelectItem
                        key={credential.credential_name}
                        value={credential.credential_name}
                      >
                        {credential.credential_name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            />
          </div>

          <div className="space-y-1">
            <LabelWithTooltip
              htmlFor="vs-metadata"
              tooltip="JSON metadata for the vector store (optional)"
            >
              Metadata
            </LabelWithTooltip>
            <Textarea
              id="vs-metadata"
              rows={4}
              value={metadataJson}
              onChange={(e) => setMetadataJson(e.target.value)}
              placeholder='{"key": "value"}'
            />
          </div>

          <DialogFooter>
            <Button type="button" variant="outline" onClick={handleClose}>
              Cancel
            </Button>
            <Button type="submit">Create</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

export default VectorStoreForm;
