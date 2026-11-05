import React, { useState, useEffect } from "react";
import type { ParseKeys } from "i18next";
import { Trans, useTranslation } from "react-i18next";
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
import i18n from "@/i18n/i18n";
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

const EMBEDDING_MODEL_RENAME_PROVIDERS = new Set(["milvus", "valkey", "mongodb", "opensearch"]);

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
  "opensearch_embedding_field",
  "opensearch_text_field",
  "opensearch_index_name",
] as const;

type ProviderFieldName = (typeof PROVIDER_FIELD_NAMES)[number];

const isProviderFieldName = (name: string): name is ProviderFieldName =>
  (PROVIDER_FIELD_NAMES as readonly string[]).includes(name);

const optionalText = z.string().optional();

const vectorStoreShape = {
  custom_llm_provider: z.string().min(1, { error: () => i18n.t("vectorStores.form.errors.providerRequired") }),
  vector_store_id: z.string().min(1, { error: () => i18n.t("vectorStores.form.errors.vectorStoreIdRequired") }),
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
  opensearch_embedding_field: optionalText,
  opensearch_text_field: optionalText,
  opensearch_index_name: optionalText,
};

const vectorStoreSchema = z.object(vectorStoreShape).superRefine((values, ctx) => {
  getProviderSpecificFields(values.custom_llm_provider)
    .filter((field) => field.required && isProviderFieldName(field.name) && !values[field.name])
    .forEach((field) =>
      ctx.addIssue({
        code: "custom",
        path: [field.name],
        message:
          field.type === "select"
            ? i18n.t("vectorStores.form.errors.selectField", { field: i18n.t(field.label).toLowerCase() })
            : i18n.t("vectorStores.form.errors.inputField", { field: i18n.t(field.label).toLowerCase() }),
      }),
    );
});

type VectorStoreFormValues = z.output<typeof vectorStoreSchema>;

const VECTOR_STORE_ID_PLACEHOLDERS: Record<string, ParseKeys> = {
  vertex_rag_engine: "vectorStores.form.idPlaceholders.vertexRagEngine",
  "vertex_ai/search_api": "vectorStores.form.idPlaceholders.vertexAiSearch",
  valkey: "vectorStores.form.idPlaceholders.valkey",
  mongodb: "vectorStores.form.idPlaceholders.mongodb",
  opensearch: "vectorStores.form.idPlaceholders.opensearch",
};

const VERTEX_SEARCH_API_WITH_ENGINE_PLACEHOLDER: ParseKeys =
  "vectorStores.form.idPlaceholders.vertexAiSearchWithEngine";

const DEFAULT_VECTOR_STORE_ID_PLACEHOLDER: ParseKeys = "vectorStores.form.idPlaceholders.default";

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
  opensearch_embedding_field: "embedding",
  opensearch_text_field: "text",
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
    const { t } = useTranslation();
    const [revealed, setRevealed] = useState(false);
    return (
      <InputGroup>
        <InputGroupInput {...props} ref={ref} type={revealed ? "text" : "password"} />
        <InputGroupAddon align="inline-end">
          <InputGroupButton
            size="icon-xs"
            aria-label={revealed ? t("vectorStores.form.password.hide") : t("vectorStores.form.password.show")}
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
  const { t } = useTranslation();
  const form = useZodForm(vectorStoreSchema, { defaultValues: EMPTY_VALUES });
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
        toast.fromError(t("vectorStores.form.toasts.invalidMetadata"));
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
      toast.success(t("vectorStores.form.toasts.created"));
      form.reset(EMPTY_VALUES);
      setMetadataJson("{}");
      onSuccess();
    } catch (error) {
      console.error("Error creating vector store:", error);
      toast.fromError(t("vectorStores.form.toasts.createFailed", { error: String(error) }));
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
          <DialogTitle>{t("vectorStores.form.title")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={form.handleSubmit(handleCreate)}>
            <FieldGroup>
              <FormField
                control={form.control}
                name="custom_llm_provider"
                label={labelWithHint(
                  t("vectorStores.form.fields.provider"),
                  t("vectorStores.form.fields.providerHint"),
                )}
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
                  <AlertTitle>{t("vectorStores.form.pgVector.title")}</AlertTitle>
                  <AlertDescription>
                    <p>{t("vectorStores.form.pgVector.intro")}</p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        {t("vectorStores.form.pgVector.steps.deploy")}{" "}
                        <a href="https://github.com/BerriAI/litellm-pgvector" target="_blank" rel="noopener noreferrer">
                          https://github.com/BerriAI/litellm-pgvector
                        </a>
                      </li>
                      <li>{t("vectorStores.form.pgVector.steps.configure")}</li>
                      <li>{t("vectorStores.form.pgVector.steps.start")}</li>
                      <li>{t("vectorStores.form.pgVector.steps.enter")}</li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              {selectedProvider === "valkey" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>{t("vectorStores.form.valkey.title")}</AlertTitle>
                  <AlertDescription>
                    <p>{t("vectorStores.form.valkey.intro")}</p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>{t("vectorStores.form.valkey.steps.searchEnabled")}</li>
                      <li>{t("vectorStores.form.valkey.steps.indexLoaded")}</li>
                      <li>{t("vectorStores.form.valkey.steps.embeddingModel")}</li>
                      <li>{t("vectorStores.form.valkey.steps.fieldNames")}</li>
                    </ol>
                    <p style={{ marginTop: "8px" }}>{t("vectorStores.form.valkey.howItWorks")}</p>
                  </AlertDescription>
                </Alert>
              )}

              {selectedProvider === "opensearch" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>{t("vectorStores.form.opensearch.title")}</AlertTitle>
                  <AlertDescription>
                    <p>{t("vectorStores.form.opensearch.intro")}</p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>{t("vectorStores.form.opensearch.steps.knnField")}</li>
                      <li>{t("vectorStores.form.opensearch.steps.embeddingModel")}</li>
                      <li>{t("vectorStores.form.opensearch.steps.credentials")}</li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              {selectedProvider === "vertex_rag_engine" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>{t("vectorStores.form.vertexRagEngine.title")}</AlertTitle>
                  <AlertDescription>
                    <p>{t("vectorStores.form.vertexRagEngine.intro")}</p>
                    <p style={{ marginTop: "4px", fontStyle: "italic" }}>
                      {t("vectorStores.form.vertexRagEngine.renameNote")}
                    </p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        <Trans
                          i18nKey="vectorStores.form.vertexRagEngine.steps.guide"
                          components={{
                            guide: (
                              <a
                                href="https://cloud.google.com/vertex-ai/generative-ai/docs/rag-engine/rag-overview"
                                target="_blank"
                                rel="noopener noreferrer"
                              />
                            ),
                          }}
                        />
                      </li>
                      <li>{t("vectorStores.form.vertexRagEngine.steps.createCorpus")}</li>
                      <li>{t("vectorStores.form.vertexRagEngine.steps.noteCorpusId")}</li>
                      <li>{t("vectorStores.form.vertexRagEngine.steps.enterCorpusId")}</li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              {selectedProvider === "vertex_ai/search_api" && (
                <Alert variant="info">
                  <Info />
                  <AlertTitle>{t("vectorStores.form.vertexAiSearch.title")}</AlertTitle>
                  <AlertDescription>
                    <p>{t("vectorStores.form.vertexAiSearch.intro")}</p>
                    <p style={{ marginTop: "4px", fontStyle: "italic" }}>
                      {t("vectorStores.form.vertexAiSearch.renameNote")}
                    </p>
                    <ol style={{ marginLeft: "16px", marginTop: "8px", listStyleType: "decimal" }}>
                      <li>
                        <Trans
                          i18nKey="vectorStores.form.vertexAiSearch.steps.guide"
                          components={{
                            guide: (
                              <a
                                href="https://cloud.google.com/generative-ai-app-builder/docs/create-data-store-es"
                                target="_blank"
                                rel="noopener noreferrer"
                                style={{ textDecoration: "underline" }}
                              />
                            ),
                          }}
                        />
                      </li>
                      <li>{t("vectorStores.form.vertexAiSearch.steps.location")}</li>
                      <li>{t("vectorStores.form.vertexAiSearch.steps.dataStoreId")}</li>
                      <li>
                        <Trans
                          i18nKey="vectorStores.form.vertexAiSearch.steps.engineId"
                          components={{ strong: <strong /> }}
                        />
                      </li>
                    </ol>
                  </AlertDescription>
                </Alert>
              )}

              <FormField
                control={form.control}
                name="vector_store_id"
                label={labelWithHint(
                  t("vectorStores.form.fields.vectorStoreId"),
                  t("vectorStores.form.fields.vectorStoreIdHint"),
                )}
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
                label={labelWithHint(t("vectorStores.form.fields.name"), t("vectorStores.form.fields.nameHint"))}
              >
                {({ ref, value, ...field }) => <Input {...field} ref={ref} value={value ?? ""} />}
              </FormField>

              <FormField
                control={form.control}
                name="vector_store_description"
                label={t("vectorStores.form.fields.description")}
              >
                {({ ref, value, ...field }) => <Textarea {...field} ref={ref} value={value ?? ""} rows={4} />}
              </FormField>

              <FormField
                control={form.control}
                name="litellm_credential_name"
                label={labelWithHint(
                  t("vectorStores.form.fields.credentials"),
                  t("vectorStores.form.fields.credentialsHint"),
                )}
              >
                {({ id, value, onChange }) => (
                  <SearchSelect
                    inputId={id}
                    placeholder={t("vectorStores.form.fields.credentialsPlaceholder")}
                    options={credentialOptions(credentials)}
                    value={value ?? ""}
                    onValueChange={(selected) => onChange(selected === "" || selected === null ? undefined : selected)}
                  />
                )}
              </FormField>

              <div role="group" className="flex w-full flex-col gap-3">
                <span className="flex w-fit gap-2 text-sm leading-snug font-medium">
                  {labelWithHint(t("vectorStores.form.fields.metadata"), t("vectorStores.form.fields.metadataHint"))}
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
                {t("vectorStores.form.buttons.cancel")}
              </Button>
              <Button type="submit">{t("vectorStores.form.buttons.create")}</Button>
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
  const { t } = useTranslation();
  const label = labelWithHint(t(field.label), t(field.tooltip));

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
      <FormField control={control} name={field.name} label={label}>
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
              <ComboboxEmpty>{t("vectorStores.form.noMatchingOptions")}</ComboboxEmpty>
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
    <FormField control={control} name={field.name} label={label}>
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
