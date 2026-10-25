import { useLanguage } from "@/contexts/LanguageContext";
import React, { useState, useEffect, useMemo } from "react";
import { englishTranslate, type Translate } from "@/i18n/language";
import { CircleHelp, Eye, EyeOff, Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { useWatch } from "react-hook-form";
import { z } from "zod";
import { CredentialItem, vectorStoreCreateCall } from "@/components/networking";
import {
  VectorStoreProviders,
  vectorStoreProviderLogoMap,
  vectorStoreProviderMap,
  getProviderSpecificFields,
  getVectorStoreProviderLogoAndName,
  VectorStoreFieldConfig,
} from "@/components/vector_store_providers";
import { Logo } from "@/components/molecules/logo/Logo";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { toast } from "@/lib/toast";
import { extractProxyErrorMessage } from "@/lib/http/client";
import TranslatedMessage from "@/components/shared/TranslatedMessage";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { useZodForm } from "@/lib/forms/useZodForm";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { credentialOptions } from "@/components/shared/credentialOptions";

const EMBEDDING_MODEL_RENAME_PROVIDERS = new Set(["milvus", "valkey", "mongodb"]);

export const buildVectorStoreLitellmParams = (
  provider: string,
  formValues: Record<string, unknown>,
): Record<string, unknown> =>
  Object.fromEntries(
    getProviderSpecificFields(provider)
      .filter(isSupportedProviderField)
      .map((field) => [
        EMBEDDING_MODEL_RENAME_PROVIDERS.has(provider) && field.name === "embedding_model"
          ? "litellm_embedding_model"
          : field.name,
        formValues[field.name],
      ]),
  );

interface VectorStoreFormProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
  accessToken: string | null;
  credentials: CredentialItem[];
}

const PROVIDER_FIELD_NAMES = [
  "api_base",
  "api_key",
  "vertex_project",
  "vertex_location",
  "vertex_collection_id",
  "vertex_engine_id",
  "embedding_model",
  "vector_bucket_name",
  "index_name",
  "aws_region_name",
  "mongodb_database",
  "mongodb_collection",
  "mongodb_embedding_field",
  "mongodb_text_field",
  "mongodb_num_candidates",
  "valkey_host",
  "valkey_port",
  "valkey_password",
  "valkey_ssl",
  "valkey_text_field",
  "valkey_embedding_field",
] as const;

type ProviderFieldName = (typeof PROVIDER_FIELD_NAMES)[number];

const isProviderFieldName = (name: string): name is ProviderFieldName =>
  (PROVIDER_FIELD_NAMES as readonly string[]).includes(name);

const optionalText = z.string().optional();

const vectorStoreShape = (t: Translate) => ({
  custom_llm_provider: z.string().min(1, t("Please select a provider")),
  vector_store_id: z.string().min(1, t("Please input the vector store ID from your api provider")),
  vector_store_name: optionalText,
  vector_store_description: optionalText,
  litellm_credential_name: z.string().nullable().optional(),
  api_base: optionalText,
  api_key: optionalText,
  vertex_project: optionalText,
  vertex_location: optionalText,
  vertex_collection_id: optionalText,
  vertex_engine_id: optionalText,
  embedding_model: optionalText,
  vector_bucket_name: optionalText,
  index_name: optionalText,
  aws_region_name: optionalText,
  mongodb_database: optionalText,
  mongodb_collection: optionalText,
  mongodb_embedding_field: optionalText,
  mongodb_text_field: optionalText,
  mongodb_num_candidates: optionalText,
  valkey_host: optionalText,
  valkey_port: optionalText,
  valkey_password: optionalText,
  valkey_ssl: optionalText,
  valkey_text_field: optionalText,
  valkey_embedding_field: optionalText,
});

// 校验只翻译内置字段说明，不改提供商标识和提交数据。
const vectorStoreSchema = (t: Translate = englishTranslate) =>
  z.object(vectorStoreShape(t)).superRefine((values, ctx) => {
    getProviderSpecificFields(values.custom_llm_provider)
      .filter((field) => field.required && isProviderFieldName(field.name) && !values[field.name])
      .forEach((field) =>
        ctx.addIssue({
          code: "custom",
          path: [field.name],
          message:
            field.type === "select"
              ? t("Please select the {field}", {
                  field: t(field.label) === field.label ? field.label.toLowerCase() : t(field.label),
                })
              : t("Please input the {field}", {
                  field: t(field.label) === field.label ? field.label.toLowerCase() : t(field.label),
                }),
        }),
      );
  });

type VectorStoreFormValues = z.output<ReturnType<typeof vectorStoreSchema>>;

const VECTOR_STORE_ID_PLACEHOLDERS: Record<string, string> = {
  vertex_rag_engine: '6917529027641081856 (corpus ID from Vertex AI / "RAG Engine" console)',
  "vertex_ai/search_api": 'my-datastore_1234567890 (data store ID from Vertex AI / "Agent Search" console)',
  valkey: "my-search-index (FT index name in Valkey)",
  mongodb: "my-vector-index (MongoDB Vector Search index name)",
};

const VERTEX_SEARCH_API_WITH_ENGINE_PLACEHOLDER = "Any identifier you'll use to reference this in LiteLLM";

const DEFAULT_VECTOR_STORE_ID_PLACEHOLDER = "Enter vector store ID from your provider";

const EMPTY_VALUES: VectorStoreFormValues = {
  custom_llm_provider: "bedrock",
  vector_store_id: "",
  vertex_location: "global",
  mongodb_embedding_field: "embedding",
  mongodb_text_field: "text",
  valkey_port: "6379",
  valkey_ssl: "false",
  valkey_text_field: "text",
  valkey_embedding_field: "embedding",
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

const PasswordInput = React.forwardRef<HTMLInputElement, React.ComponentPropsWithoutRef<typeof InputGroupInput>>(
  (props, ref) => {
    const { t } = useLanguage();
    const [revealed, setRevealed] = useState(false);
    return (
      <InputGroup>
        <InputGroupInput {...props} ref={ref} type={revealed ? "text" : "password"} />
        <InputGroupAddon align="inline-end">
          <InputGroupButton
            size="icon-xs"
            aria-label={revealed ? t("Hide Password") : t("Show Password")}
            onClick={() => setRevealed(!revealed)}
          >
            {revealed ? <EyeOff /> : <Eye />}
          </InputGroupButton>
        </InputGroupAddon>
      </InputGroup>
    );
  },
);
PasswordInput.displayName = "PasswordInput";

const VectorStoreForm: React.FC<VectorStoreFormProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  accessToken,
  credentials,
}) => {
  const { t } = useLanguage();
  // 校验保存固定原文，已有错误只在显示处翻译，不随语言变化重新校验。
  const schema = useMemo(() => vectorStoreSchema(), []);
  const form = useZodForm(schema, { defaultValues: EMPTY_VALUES });
  const [metadataJson, setMetadataJson] = useState("{}");
  const [selectedProvider, setSelectedProvider] = useState("bedrock");
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const vertexEngineId = useWatch({ control: form.control, name: "vertex_engine_id" });

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

  const makeProviderChangeHandler = (onChange: (provider: string) => void) => (provider: string | null) => {
    if (provider === null) return;
    onChange(provider);
    setSelectedProvider(provider);
  };

  const handleCreate = async (formValues: VectorStoreFormValues) => {
    if (!accessToken) return;
    try {
      let metadata = {};
      try {
        metadata = metadataJson.trim() ? JSON.parse(metadataJson) : {};
      } catch (e) {
        toast.fromError("Invalid JSON in metadata field", {
          description: <TranslatedMessage message="Invalid JSON in metadata field" />,
        });
        return;
      }

      await vectorStoreCreateCall(accessToken, {
        vector_store_id: formValues.vector_store_id,
        custom_llm_provider: formValues.custom_llm_provider,
        vector_store_name: formValues.vector_store_name,
        vector_store_description: formValues.vector_store_description,
        vector_store_metadata: metadata,
        litellm_credential_name: formValues.litellm_credential_name,
        litellm_params: buildVectorStoreLitellmParams(formValues.custom_llm_provider, formValues),
      });
      toast.success(<TranslatedMessage message="Vector store created successfully" />);
      form.reset(EMPTY_VALUES);
      setMetadataJson("{}");
      onSuccess();
    } catch (error) {
      console.error("Error creating vector store:", error);
      // 保留原始错误的状态分类，只翻译本地上下文，不把服务端详情作为词典键。
      toast.fromError(error, {
        description: (
          <TranslatedMessage
            message="Error creating vector store: {error}"
            values={{ error: extractProxyErrorMessage(error) }}
          />
        ),
      });
    }
  };

  const handleCancel = () => {
    form.reset(EMPTY_VALUES);
    setMetadataJson("{}");
    setSelectedProvider("bedrock");
    onCancel();
  };

  const vectorStoreIdPlaceholder =
    selectedProvider === "vertex_ai/search_api" && vertexEngineId
      ? VERTEX_SEARCH_API_WITH_ENGINE_PLACEHOLDER
      : VECTOR_STORE_ID_PLACEHOLDERS[selectedProvider] ?? DEFAULT_VECTOR_STORE_ID_PLACEHOLDER;

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
        <DialogHeader>
          <DialogTitle>{t("Add New Vector Store")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={form.handleSubmit(handleCreate)}>
            <FieldGroup>
              <FormField
                control={form.control}
                name="custom_llm_provider"
                label={labelWithHint(t("Provider"), t("Select the provider for this vector store"))}
                renderError={({ message, type }) =>
                  type === "too_small" && message === "Please select a provider" ? t(message) : message
                }
              >
                {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
                  <Select value={value} onValueChange={makeProviderChangeHandler(onChange)}>
                    <SelectTrigger
                      id={id}
                      aria-invalid={ariaInvalid}
                      aria-describedby={ariaDescribedBy}
                      className="w-full"
                    >
                      <SelectValue>
                        {(provider: string) => {
                          const { displayName, logo } = getVectorStoreProviderLogoAndName(provider);
                          return (
                            <>
                              <Logo src={logo} label={displayName} className="w-5 h-5" />
                              <span>{displayName}</span>
                            </>
                          );
                        }}
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent>
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
                )}
              </FormField>

              {selectedProvider === "pg_vector" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>{t("PG Vector Setup Required")}</AlertTitle>
                  <AlertDescription>
                    <p>{t("LiteLLM provides a server to connect to PG Vector. To use this provider:")}</p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        {t("Deploy the litellm-pgvector server from:")}{" "}
                        <a href="https://github.com/BerriAI/litellm-pgvector" target="_blank" rel="noopener noreferrer">
                          https://github.com/BerriAI/litellm-pgvector
                        </a>
                      </li>
                      <li>{t("Configure your PostgreSQL database with pgvector extension")}</li>
                      <li>{t("Start the server and note the API base URL and API key")}</li>
                      <li>{t("Enter those details in the fields below")}</li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              {selectedProvider === "valkey" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>{t("Valkey Setup Required")}</AlertTitle>
                  <AlertDescription>
                    <p>
                      {t(
                        "LiteLLM searches documents you have already stored in Valkey. It does not create the index or upload documents for you. Before creating this vector store, make sure:",
                      )}
                    </p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        {t(
                          "Your Valkey server has vector search enabled (the valkey-search module, included in the valkey-bundle image and in AWS ElastiCache / MemoryDB for Valkey)",
                        )}
                      </li>
                      <li>
                        {t(
                          "You have already created a search index and loaded your documents and their embeddings into it. Enter that index name as the Vector Store ID",
                        )}
                      </li>
                      <li>
                        {t(
                          "You know which embedding model created those stored embeddings. That model must be added to this proxy under Models so you can pick it below. Using a different model returns wrong results",
                        )}
                      </li>
                      <li>
                        {t(
                          'You know the field names your documents use for their text and their embedding. If they are not "text" and "embedding", set them below',
                        )}
                      </li>
                    </ol>
                    <p style={{ marginTop: "8px" }}>
                      {t(
                        "When a query comes in, LiteLLM converts it to an embedding with the model below and returns the closest matching documents from your index.",
                      )}
                    </p>
                  </AlertDescription>
                </Alert>
              )}

              {selectedProvider === "vertex_rag_engine" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>{t("Vertex AI RAG Engine Setup")}</AlertTitle>
                  <AlertDescription>
                    <p>{t("To use Vertex AI RAG Engine:")}</p>
                    <p style={{ marginTop: "4px", fontStyle: "italic" }}>
                      {t(
                        'Note: Google Cloud has renamed this to "RAG Engine" in its console — the steps below still apply.',
                      )}
                    </p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        {t("Set up your Vertex AI RAG Engine corpus following the guide:")}{" "}
                        <a
                          href="https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          {t("Vertex AI RAG Engine Overview")}
                        </a>
                      </li>
                      <li>{t("Create a corpus in your Google Cloud project")}</li>
                      <li>
                        {t('Note the corpus ID from the Vertex AI console (now labeled "RAG Engine" in Google Cloud)')}
                      </li>
                      <li>{t("Enter the corpus ID in the Vector Store ID field below")}</li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              {selectedProvider === "vertex_ai/search_api" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>{t("Vertex AI Search Setup")}</AlertTitle>
                  <AlertDescription>
                    <p>{t("To use Vertex AI Search (Discovery Engine):")}</p>
                    <p style={{ marginTop: "4px", fontStyle: "italic" }}>
                      {t(
                        'Note: Google Cloud has renamed this to "Agent Search" in its console — the steps below still apply.',
                      )}
                    </p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        {t(
                          "Enable the Discovery Engine API on your Google Cloud project and create a data store following the guide:",
                        )}{" "}
                        <a
                          href="https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es"
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ textDecoration: "underline" }}
                        >
                          {t("Create a Vertex AI Search data store")}
                        </a>
                      </li>
                      <li>{t("Pick a supported location: global, us, or eu")}</li>
                      <li>
                        {t(
                          "For most data store types (Cloud Storage, BigQuery, Media): copy the data store ID and enter it in the Vector Store ID field below.",
                        )}
                      </li>
                      <li>
                        {t(
                          "For website, healthcare, and connector-based sources (Drive, Gmail, Slack, Jira, etc.): create a search app on top of the data store, then copy the",
                        )}{" "}
                        <strong>{t("Engine ID")}</strong>{" "}
                        {t(
                          "and enter it in the Engine ID field. The Vector Store ID is still required as the LiteLLM-side name for this record, but it isn't used in the GCP URL when Engine ID is set.",
                        )}
                      </li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              <FormField
                control={form.control}
                name="vector_store_id"
                label={labelWithHint(t("Vector Store ID"), t("Enter the vector store ID from your api provider"))}
                renderError={({ message, type }) =>
                  type === "too_small" && message === "Please input the vector store ID from your api provider"
                    ? t(message)
                    : message
                }
              >
                {({ ref, ...field }) => <Input {...field} ref={ref} placeholder={t(vectorStoreIdPlaceholder)} />}
              </FormField>

              {getProviderSpecificFields(selectedProvider)
                .filter(isSupportedProviderField)
                .map((field) => (
                  <ProviderField key={field.name} field={field} control={form.control} modelInfo={modelInfo} />
                ))}

              <FormField
                control={form.control}
                name="vector_store_name"
                label={labelWithHint(
                  t("Vector Store Name"),
                  t("Custom name you want to give to the vector store, this name will be rendered on the LiteLLM UI"),
                )}
              >
                {({ ref, value, ...field }) => <Input {...field} ref={ref} value={value ?? ""} />}
              </FormField>

              <FormField control={form.control} name="vector_store_description" label={t("Description")}>
                {({ ref, value, ...field }) => <Textarea {...field} ref={ref} value={value ?? ""} rows={4} />}
              </FormField>

              <FormField
                control={form.control}
                name="litellm_credential_name"
                label={labelWithHint(
                  t("Existing Credentials"),
                  t("Optionally select API provider credentials for this vector store eg. Bedrock API KEY"),
                )}
              >
                {({ id, value, onChange }) => (
                  <SearchSelect
                    inputId={id}
                    placeholder={t("Select or search for existing credentials")}
                    options={credentialOptions(credentials)}
                    value={value ?? ""}
                    onValueChange={(selected) => onChange(selected === "" || selected === null ? undefined : selected)}
                  />
                )}
              </FormField>

              <div role="group" className="flex w-full flex-col gap-3">
                <span className="flex w-fit gap-2 text-sm leading-snug font-medium">
                  {labelWithHint(t("Metadata"), t("JSON metadata for the vector store (optional)"))}
                </span>
                <Textarea
                  rows={4}
                  value={metadataJson}
                  onChange={(event) => setMetadataJson(event.target.value)}
                  placeholder='{"key": "value"}'
                />
              </div>
            </FieldGroup>

            <div className="mt-6 flex justify-end space-x-3">
              <Button type="button" variant="outline" onClick={handleCancel}>
                {t("Cancel")}
              </Button>
              <Button type="submit">{t("Create")}</Button>
            </div>
          </form>
        </TooltipProvider>
      </DialogContent>
    </Dialog>
  );
};

type SupportedProviderField = VectorStoreFieldConfig & { name: ProviderFieldName };

const isSupportedProviderField = (field: VectorStoreFieldConfig): field is SupportedProviderField =>
  isProviderFieldName(field.name);

interface ProviderFieldProps {
  field: SupportedProviderField;
  control: ReturnType<typeof useZodForm<VectorStoreFormValues, VectorStoreFormValues>>["control"];
  modelInfo: ModelGroup[];
}

const ProviderField: React.FC<ProviderFieldProps> = ({ field, control, modelInfo }) => {
  const { t } = useLanguage();
  // 字段来自内置提供商定义，模型选项与最终提交的字段名保持原样。
  const label = labelWithHint(t(field.label), t(field.tooltip));
  const validationMessage = field.type === "select" ? "Please select the {field}" : "Please input the {field}";
  const renderError = ({ message, type }: { message?: string; type: string }) =>
    type === "custom" && message === englishTranslate(validationMessage, { field: field.label.toLowerCase() })
      ? t(validationMessage, {
          field: t(field.label) === field.label ? field.label.toLowerCase() : t(field.label),
        })
      : message;

  if (field.type === "select") {
    const selectOptions =
      field.options ??
      modelInfo
        .filter((option: ModelGroup) => option.mode === "embedding" || option.mode === null)
        .map((option: ModelGroup) => ({
          value: option.model_group,
          label: option.model_group,
        }));

    return (
      <FormField control={control} name={field.name} label={label} renderError={renderError}>
        {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": ariaDescribedBy }) => (
          <Combobox
            items={selectOptions}
            value={selectOptions.find((option) => option.value === value) ?? null}
            onValueChange={(option: { value: string; label: string } | null) => onChange(option?.value)}
            itemToStringLabel={(option: { value: string; label: string }) => option.label}
            isItemEqualToValue={(
              option: { value: string; label: string },
              selected: { value: string; label: string },
            ) => option.value === selected.value}
          >
            <ComboboxInput
              id={id}
              aria-invalid={ariaInvalid}
              aria-describedby={ariaDescribedBy}
              placeholder={field.placeholder}
              className="w-full"
            />
            <ComboboxContent>
              <ComboboxEmpty>{t("No matching options")}</ComboboxEmpty>
              <ComboboxList>
                {(option: { value: string; label: string }) => (
                  <ComboboxItem key={option.value} value={option}>
                    {option.label}
                  </ComboboxItem>
                )}
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        )}
      </FormField>
    );
  }

  return (
    <FormField control={control} name={field.name} label={label} renderError={renderError}>
      {({ ref, value, ...controlProps }) =>
        field.type === "password" ? (
          <PasswordInput {...controlProps} ref={ref} value={value ?? ""} placeholder={field.placeholder} />
        ) : (
          <Input {...controlProps} ref={ref} value={value ?? ""} type="text" placeholder={field.placeholder} />
        )
      }
    </FormField>
  );
};

export default VectorStoreForm;
