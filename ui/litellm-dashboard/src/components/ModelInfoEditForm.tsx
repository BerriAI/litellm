"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { CircleHelp } from "lucide-react";
import type { Dayjs } from "dayjs";
import * as React from "react";
import { useForm, type Resolver } from "react-hook-form";
import { z } from "zod/v4";

import { TagsInput } from "@/app/(dashboard)/guardrails/_components/content_filter/TagsInput";
import { FormField } from "@/components/shared/form/FormField";
import { UtcDateTimeInput } from "@/components/shared/form/UtcDateTimeInput";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

import CacheControlInjectionPoints, {
  CACHE_CONTROL_LABEL,
  CACHE_CONTROL_TOOLTIP,
  type CacheControlInjectionPoint,
} from "./add_model/cache_control_settings";
import type { CredentialItem } from "./networking";
import NumericalInput from "./shared/numerical_input";
import type { Tag } from "./tag_management/types";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import { formatPtuUtcDisplay, utcIsoToPickerValue } from "../utils/ptuDatetime";
import { isMaskedSecret } from "../utils/maskedSecretUtils";
import {
  MAX_COST_PER_PTU_PER_HOUR,
  MAX_PTU_COUNT,
  PTU_COUNT_FIELD,
  PTU_END_FIELD,
  PTU_RATE_FIELD,
  PTU_START_FIELD,
  isFilledPtuValue,
  isNonNegativePtuRate,
  isPositiveWholePtuCount,
  ptuWindowIsOrdered,
} from "../utils/ptuValidation";

interface PtuEditField {
  name: string;
  label: string;
  input: "number" | "datetime";
  placeholder?: string;
  isCount?: boolean;
}

const PTU_EDIT_FIELDS: PtuEditField[] = [
  { name: PTU_COUNT_FIELD, label: "PTU Count", input: "number", placeholder: "e.g. 15", isCount: true },
  { name: PTU_RATE_FIELD, label: "Cost per PTU / Hour (USD)", input: "number", placeholder: "e.g. 2.00" },
  { name: PTU_START_FIELD, label: "PTU Effective From (UTC)", input: "datetime" },
  { name: PTU_END_FIELD, label: "PTU Effective To (UTC)", input: "datetime" },
];

export type TouchedPricingField = "input_cost" | "output_cost" | "cache_read_cost" | "cache_write_cost";

const PRICING_FIELDS: readonly TouchedPricingField[] = [
  "input_cost",
  "output_cost",
  "cache_read_cost",
  "cache_write_cost",
] as const;

const COST_SOURCES: Record<TouchedPricingField, { param: string; info: string }> = {
  input_cost: { param: "input_cost_per_token", info: "input_cost_per_token" },
  output_cost: { param: "output_cost_per_token", info: "output_cost_per_token" },
  cache_read_cost: { param: "cache_read_input_token_cost", info: "cache_read_input_token_cost" },
  cache_write_cost: { param: "cache_creation_input_token_cost", info: "cache_creation_input_token_cost" },
};

export interface ModelEditFormValues {
  model_name?: string;
  litellm_model_name?: string;
  api_base?: string;
  custom_llm_provider?: string;
  organization?: string;
  tpm?: string | number | null;
  rpm?: string | number | null;
  max_retries?: string | number | null;
  timeout?: string | number | null;
  stream_timeout?: string | number | null;
  input_cost?: string | number | null;
  output_cost?: string | number | null;
  cache_read_cost?: string | number | null;
  cache_write_cost?: string | number | null;
  ptu_count?: string | number | null;
  cost_per_ptu_per_hour?: string | number | null;
  ptu_effective_from?: Dayjs | null;
  ptu_effective_to?: Dayjs | null;
  cache_control?: boolean;
  cache_control_injection_points?: CacheControlInjectionPoint[];
  model_access_group?: string[];
  guardrails?: string[];
  vector_store_ids?: string[];
  tags?: string[];
  health_check_model?: string | null;
  litellm_credential_name?: string;
  litellm_extra_params?: string;
  model_info?: string;
}

type ModelEditFieldName = keyof ModelEditFormValues;

const scalar = z.union([z.string(), z.number(), z.null()]).optional();
const textish = z.string().optional();

const modelEditShape = {
  model_name: textish,
  litellm_model_name: textish,
  api_base: textish,
  custom_llm_provider: textish,
  organization: textish,
  tpm: scalar,
  rpm: scalar,
  max_retries: scalar,
  timeout: scalar,
  stream_timeout: scalar,
  input_cost: scalar,
  output_cost: scalar,
  cache_read_cost: scalar,
  cache_write_cost: scalar,
  ptu_count: scalar,
  cost_per_ptu_per_hour: scalar,
  ptu_effective_from: z.custom<Dayjs | null>().nullish(),
  ptu_effective_to: z.custom<Dayjs | null>().nullish(),
  cache_control: z.boolean().optional(),
  cache_control_injection_points: z.array(z.custom<CacheControlInjectionPoint>()).optional(),
  model_access_group: z.array(z.string()).optional(),
  guardrails: z.array(z.string()).optional(),
  vector_store_ids: z.array(z.string()).optional(),
  tags: z.array(z.string()).optional(),
  health_check_model: z.string().nullish(),
  litellm_credential_name: textish,
  litellm_extra_params: textish,
  model_info: textish,
};

const isJson = (value: string): boolean => {
  try {
    JSON.parse(value);
    return true;
  } catch {
    return false;
  }
};

const buildSchema = (ptuEnabled: boolean, isFieldTouched: (field: TouchedPricingField) => boolean) =>
  z.object(modelEditShape).superRefine((values, ctx) => {
    const reject = (path: ModelEditFieldName, message: string) =>
      ctx.addIssue({ code: "custom", path: [path], message });

    if (values.litellm_extra_params && !isJson(values.litellm_extra_params)) {
      reject("litellm_extra_params", "Please enter valid JSON");
    }

    // antd validates only mounted fields, and the PTU block does not render when the flag is off.
    if (!ptuEnabled) {
      return;
    }

    if (!isPositiveWholePtuCount(values.ptu_count)) {
      reject("ptu_count", `PTU Count must be a whole number between 1 and ${MAX_PTU_COUNT.toLocaleString()}`);
    }
    if (!isNonNegativePtuRate(values.cost_per_ptu_per_hour)) {
      reject(
        "cost_per_ptu_per_hour",
        `Cost per PTU / Hour must be between 0 and ${MAX_COST_PER_PTU_PER_HOUR.toLocaleString()}`,
      );
    }
    if (isFilledPtuValue(values.ptu_count) !== isFilledPtuValue(values.cost_per_ptu_per_hour)) {
      const message = "PTU Count and Cost per PTU / Hour must be set together";
      reject("ptu_count", message);
      reject("cost_per_ptu_per_hour", message);
    }
    if (isFilledPtuValue(values.ptu_count) && !isFilledPtuValue(values.ptu_effective_from)) {
      reject("ptu_effective_from", "PTU Effective From is required when PTU Count is set");
    }
    if (!ptuWindowIsOrdered(values.ptu_effective_from, values.ptu_effective_to)) {
      const message = "PTU Effective To must be after PTU Effective From";
      reject("ptu_effective_from", message);
      reject("ptu_effective_to", message);
    }

    for (const field of PRICING_FIELDS) {
      const value = values[field];
      if (
        isFieldTouched(field) &&
        isFilledPtuValue(values.ptu_count) &&
        isFilledPtuValue(value) &&
        Number(value) !== 0
      ) {
        reject(field, "A PTU deployment bills by reserved capacity, so this cost must be 0 or blank");
      }
    }
  });

const perMillionTokens = (...rates: (number | null | undefined)[]): number | null => {
  const rate = rates.find((candidate) => candidate != null);
  return rate == null ? null : rate * 1_000_000;
};

export const toModelEditFormValues = (localModelData: any, isWildcardModel: boolean): ModelEditFormValues => ({
  model_name: localModelData.model_name,
  litellm_model_name: localModelData.litellm_model_name,
  api_base: localModelData.litellm_params.api_base,
  custom_llm_provider: localModelData.litellm_params.custom_llm_provider,
  organization: localModelData.litellm_params.organization,
  tpm: localModelData.litellm_params.tpm,
  rpm: localModelData.litellm_params.rpm,
  max_retries: localModelData.litellm_params.max_retries,
  timeout: localModelData.litellm_params.timeout,
  stream_timeout: localModelData.litellm_params.stream_timeout,
  input_cost: perMillionTokens(
    localModelData.litellm_params.input_cost_per_token,
    localModelData.model_info?.input_cost_per_token,
  ),
  output_cost: perMillionTokens(
    localModelData.litellm_params?.output_cost_per_token,
    localModelData.model_info?.output_cost_per_token,
  ),
  ptu_count: localModelData.model_info?.ptu_count ?? null,
  cost_per_ptu_per_hour: localModelData.model_info?.cost_per_ptu_per_hour ?? null,
  ptu_effective_from: utcIsoToPickerValue(localModelData.model_info?.ptu_effective_from),
  ptu_effective_to: utcIsoToPickerValue(localModelData.model_info?.ptu_effective_to),
  cache_read_cost: perMillionTokens(
    localModelData.litellm_params?.cache_read_input_token_cost,
    localModelData.model_info?.cache_read_input_token_cost,
  ),
  cache_write_cost: perMillionTokens(
    localModelData.litellm_params?.cache_creation_input_token_cost,
    localModelData.model_info?.cache_creation_input_token_cost,
  ),
  cache_control: localModelData.litellm_params?.cache_control_injection_points ? true : false,
  cache_control_injection_points: localModelData.litellm_params?.cache_control_injection_points || [],
  model_access_group: Array.isArray(localModelData.model_info?.access_groups)
    ? localModelData.model_info.access_groups
    : [],
  guardrails: Array.isArray(localModelData.litellm_params?.guardrails) ? localModelData.litellm_params.guardrails : [],
  vector_store_ids:
    Array.isArray(localModelData.litellm_params?.vector_store_ids) &&
    localModelData.litellm_params.vector_store_ids.length > 0
      ? localModelData.litellm_params.vector_store_ids
      : undefined,
  tags: Array.isArray(localModelData.litellm_params?.tags) ? localModelData.litellm_params.tags : [],
  // antd never mounted this field for a non-wildcard model, so the key must be absent, not null.
  ...(isWildcardModel ? { health_check_model: localModelData.model_info?.health_check_model } : {}),
  litellm_credential_name: localModelData.litellm_params?.litellm_credential_name || "",
  litellm_extra_params: JSON.stringify(
    Object.fromEntries(
      Object.entries(localModelData.litellm_params || {}).filter(
        ([key, value]) => key !== "litellm_credential_name" && !isMaskedSecret(value),
      ),
    ),
    null,
    2,
  ),
});

const displayCost = (localModelData: any, field: TouchedPricingField): string => {
  const { param, info } = COST_SOURCES[field];
  const rate = localModelData?.litellm_params?.[param] ?? localModelData?.model_info?.[info];
  return rate != null ? (Number(rate) * 1_000_000).toFixed(4) : "Not Set";
};

interface ModelInfoEditFormProps {
  localModelData: any;
  modelData: { model_info: { team_id?: string | null } & Record<string, unknown> };
  accessToken: string | null;
  isEditing: boolean;
  isSaving: boolean;
  isWildcardModel: boolean;
  ptuCostAttributionEnabled: boolean;
  showCacheControl: boolean;
  setShowCacheControl: (checked: boolean) => void;
  onCancel: () => void;
  onSubmit: (values: ModelEditFormValues, isFieldTouched: (field: TouchedPricingField) => boolean) => Promise<void>;
  modelAccessGroups: string[] | null;
  guardrailsList: string[];
  tagsList: Record<string, Tag>;
  credentialsList: CredentialItem[];
  healthCheckModelOptions: { value: string; label: string }[];
}

const Display: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="mt-1 rounded-sm bg-muted p-2">{children}</div>
);

const FIELD_LABEL_CLASS = "text-sm font-medium text-foreground";

const FieldLabel: React.FC<{ htmlFor?: string; children: React.ReactNode }> = ({ htmlFor, children }) =>
  htmlFor === undefined ? (
    <p className={FIELD_LABEL_CLASS}>{children}</p>
  ) : (
    <label htmlFor={htmlFor} className={FIELD_LABEL_CLASS}>
      {children}
    </label>
  );

const Hint: React.FC<{ text: string }> = ({ text }) => (
  <Tooltip>
    <TooltipTrigger
      render={<CircleHelp className="ml-1 inline size-3.5 shrink-0 cursor-help text-muted-foreground" />}
    />
    <TooltipContent className="max-w-xs">{text}</TooltipContent>
  </Tooltip>
);

const DocsHint: React.FC<{ text: string; href: string }> = ({ text, href }) => (
  <a href={href} target="_blank" rel="noopener noreferrer" onClick={(event) => event.stopPropagation()}>
    <Hint text={text} />
  </a>
);

const ChipList: React.FC<{ values: unknown; emptyLabel: string }> = ({ values, emptyLabel }) => {
  if (!values) {
    return <>Not Set</>;
  }
  if (!Array.isArray(values)) {
    return <>{String(values)}</>;
  }
  if (values.length === 0) {
    return <>{emptyLabel}</>;
  }
  return (
    <div className="flex flex-wrap gap-1">
      {values.map((entry: string, index: number) => (
        <Badge key={index} variant="secondary">
          {entry}
        </Badge>
      ))}
    </div>
  );
};

const ModelInfoEditForm: React.FC<ModelInfoEditFormProps> = ({
  localModelData,
  modelData,
  accessToken,
  isEditing,
  isSaving,
  isWildcardModel,
  ptuCostAttributionEnabled,
  showCacheControl,
  setShowCacheControl,
  onCancel,
  onSubmit,
  modelAccessGroups,
  guardrailsList,
  tagsList,
  credentialsList,
  healthCheckModelOptions,
}) => {
  // Neither RHF's blur-based touchedFields nor its resettable dirtyFields matches antd's touched-on-change.
  const touchedRef = React.useRef<ReadonlySet<string>>(new Set<string>());
  const isFieldTouched = React.useCallback((field: TouchedPricingField) => touchedRef.current.has(field), []);
  const markTouched = (field: string) => {
    touchedRef.current = new Set([...touchedRef.current, field]);
  };

  // react-hook-form refreshes control._options every render, so this rebuild is what the next submit runs.
  const resolver: Resolver<ModelEditFormValues> = (values, context, options) =>
    zodResolver(buildSchema(ptuCostAttributionEnabled, isFieldTouched))(values, context, options);

  const form = useForm<ModelEditFormValues>({
    resolver,
    defaultValues: toModelEditFormValues(localModelData, isWildcardModel),
  });

  const submit = (event: React.FormEvent<HTMLFormElement>) =>
    form.handleSubmit(async (values) => {
      await onSubmit(values, isFieldTouched);
    })(event);

  const cancel = () => {
    form.reset(toModelEditFormValues(localModelData, isWildcardModel));
    touchedRef.current = new Set<string>();
    onCancel();
  };

  const textField = (name: ModelEditFieldName, label: string, placeholder: string, stored: unknown) => (
    <div>
      <FieldLabel>{label}</FieldLabel>
      {isEditing ? (
        <FormField control={form.control} name={name}>
          {({ value, ...control }) => <Input {...control} value={(value as string) ?? ""} placeholder={placeholder} />}
        </FormField>
      ) : (
        <Display>{(stored as string) || "Not Set"}</Display>
      )}
    </div>
  );

  const numberField = (name: ModelEditFieldName, label: string, placeholder: string, stored: unknown) => (
    <div>
      <FieldLabel>{label}</FieldLabel>
      {isEditing ? (
        <FormField control={form.control} name={name}>
          {({ value, ...control }) => <NumericalInput {...control} value={value ?? ""} placeholder={placeholder} />}
        </FormField>
      ) : (
        <Display>{(stored as string) || "Not Set"}</Display>
      )}
    </div>
  );

  const pricingField = (name: TouchedPricingField, label: string, placeholder: string, description?: string) =>
    isEditing ? (
      <FormField control={form.control} name={name} label={label} description={description}>
        {({ value, onChange, ...control }) => (
          <NumericalInput
            {...control}
            value={value ?? ""}
            placeholder={placeholder}
            onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
              markTouched(name);
              onChange(event);
            }}
          />
        )}
      </FormField>
    ) : (
      <div>
        <FieldLabel>{label}</FieldLabel>
        <Display>{displayCost(localModelData, name)}</Display>
      </div>
    );

  const tagsField = (
    name: "model_access_group" | "guardrails" | "tags",
    options: { value: string; label: string }[],
    placeholder: string,
  ) => (
    <FormField control={form.control} name={name}>
      {({ id, value, onChange }) => (
        <TagsInput
          id={id}
          value={(value as string[]) ?? []}
          onValueChange={onChange}
          options={options}
          placeholder={placeholder}
          tokenSeparators={[","]}
        />
      )}
    </FormField>
  );

  return (
    <TooltipProvider>
      <form onSubmit={submit}>
        <div className="space-y-4">
          <div className="space-y-4">
            {textField("model_name", "Model Name", "Enter model name", localModelData.model_name)}
            {textField(
              "litellm_model_name",
              "LiteLLM Model Name",
              "Enter LiteLLM model name",
              localModelData.litellm_model_name,
            )}

            {pricingField("input_cost", "Input Cost (per 1M tokens)", "Enter input cost")}
            {pricingField("output_cost", "Output Cost (per 1M tokens)", "Enter output cost")}

            {ptuCostAttributionEnabled &&
              PTU_EDIT_FIELDS.map((ptuField) => (
                <div key={ptuField.name}>
                  <FieldLabel htmlFor={ptuField.name}>{ptuField.label}</FieldLabel>
                  {isEditing ? (
                    <FormField control={form.control} name={ptuField.name as ModelEditFieldName}>
                      {({ value, onChange, ...control }) =>
                        ptuField.input === "number" ? (
                          <NumericalInput
                            {...control}
                            id={ptuField.name}
                            onChange={onChange}
                            value={value ?? ""}
                            placeholder={ptuField.placeholder}
                            step={ptuField.isCount ? 1 : undefined}
                            min={ptuField.isCount ? 1 : 0}
                          />
                        ) : (
                          <UtcDateTimeInput
                            {...control}
                            id={ptuField.name}
                            value={value as Dayjs | null}
                            onChange={onChange}
                          />
                        )
                      }
                    </FormField>
                  ) : (
                    <Display>
                      {(ptuField.input === "datetime"
                        ? formatPtuUtcDisplay(localModelData?.model_info?.[ptuField.name])
                        : localModelData?.model_info?.[ptuField.name]) ?? "Not Set"}
                    </Display>
                  )}
                </div>
              ))}

            {pricingField(
              "cache_read_cost",
              "Cache Read Cost (per 1M tokens)",
              "Defaults to Input Cost if blank",
              "If left blank on save, defaults to Input Cost.",
            )}
            {pricingField(
              "cache_write_cost",
              "Cache Write Cost (per 1M tokens)",
              "Defaults to Input Cost if blank",
              "If left blank on save, defaults to Input Cost (backend falls back to input_cost_per_token).",
            )}

            {textField("api_base", "API Base", "Enter API base", localModelData.litellm_params?.api_base)}
            {textField(
              "custom_llm_provider",
              "Custom LLM Provider",
              "Enter custom LLM provider",
              localModelData.litellm_params?.custom_llm_provider,
            )}
            {textField(
              "organization",
              "Organization",
              "Enter organization",
              localModelData.litellm_params?.organization,
            )}

            {numberField("tpm", "TPM (Tokens per Minute)", "Enter TPM", localModelData.litellm_params?.tpm)}
            {numberField("rpm", "RPM (Requests per Minute)", "Enter RPM", localModelData.litellm_params?.rpm)}
            {numberField("max_retries", "Max Retries", "Enter max retries", localModelData.litellm_params?.max_retries)}
            {numberField("timeout", "Timeout (seconds)", "Enter timeout", localModelData.litellm_params?.timeout)}
            {numberField(
              "stream_timeout",
              "Stream Timeout (seconds)",
              "Enter stream timeout",
              localModelData.litellm_params?.stream_timeout,
            )}

            <div>
              <FieldLabel>Model Access Groups</FieldLabel>
              {isEditing ? (
                tagsField(
                  "model_access_group",
                  (modelAccessGroups ?? []).map((group) => ({ value: group, label: group })),
                  "Select existing groups or type to create new ones",
                )
              ) : (
                <Display>
                  <ChipList values={localModelData.model_info?.access_groups} emptyLabel="No groups assigned" />
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>
                Guardrails
                <DocsHint
                  text="Apply safety guardrails to this model to filter content or enforce policies"
                  href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                />
              </FieldLabel>
              {isEditing ? (
                tagsField(
                  "guardrails",
                  guardrailsList.map((name) => ({ value: name, label: name })),
                  "Select existing guardrails or type to create new ones",
                )
              ) : (
                <Display>
                  <ChipList values={localModelData.litellm_params?.guardrails} emptyLabel="No guardrails assigned" />
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>
                Attached Knowledge Bases (RAG)
                <DocsHint
                  text="Vector stores used for RAG. Every request to this model will automatically retrieve context from these knowledge bases."
                  href="https://docs.litellm.ai/docs/completion/knowledgebase"
                />
              </FieldLabel>
              {isEditing ? (
                <FormField control={form.control} name="vector_store_ids">
                  {({ value, onChange }) => (
                    <VectorStoreSelector
                      value={value as string[] | undefined}
                      onChange={onChange}
                      accessToken={accessToken || ""}
                      placeholder="Select knowledge bases (optional)"
                    />
                  )}
                </FormField>
              ) : (
                <Display>
                  <ChipList
                    values={localModelData.litellm_params?.vector_store_ids}
                    emptyLabel="No knowledge bases attached"
                  />
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>Tags</FieldLabel>
              {isEditing ? (
                tagsField(
                  "tags",
                  Object.values(tagsList).map((tag: Tag) => ({ value: tag.name, label: tag.name })),
                  "Select existing tags or type to create new ones",
                )
              ) : (
                <Display>
                  <ChipList values={localModelData.litellm_params?.tags} emptyLabel="No tags assigned" />
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>Existing Credentials</FieldLabel>
              {isEditing ? (
                <FormField control={form.control} name="litellm_credential_name">
                  {({ id, value, onChange, onBlur }) => {
                    const items = [
                      { value: "", label: "None" },
                      ...credentialsList.map((credential) => ({
                        value: credential.credential_name,
                        label: credential.credential_name,
                      })),
                    ];
                    return (
                      <Select
                        items={items}
                        value={(value as string) ?? ""}
                        onValueChange={(selected: string | null) => onChange(selected ?? "")}
                      >
                        <SelectTrigger id={id} className="w-full" onBlur={onBlur}>
                          <SelectValue placeholder="Select or search for existing credentials" />
                        </SelectTrigger>
                        <SelectContent>
                          {items.map((item) => (
                            <SelectItem key={item.value} value={item.value}>
                              {item.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    );
                  }}
                </FormField>
              ) : (
                <Display>{localModelData.litellm_params?.litellm_credential_name || "Manual"}</Display>
              )}
            </div>

            {isWildcardModel && (
              <div>
                <FieldLabel>Health Check Model</FieldLabel>
                {isEditing ? (
                  <FormField control={form.control} name="health_check_model">
                    {({ id, value, onChange, onBlur }) => (
                      <Select
                        items={healthCheckModelOptions}
                        value={(value as string | null) ?? null}
                        onValueChange={onChange}
                      >
                        <SelectTrigger id={id} className="w-full" onBlur={onBlur}>
                          <SelectValue placeholder="Select existing health check model" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value={null}>None</SelectItem>
                          {healthCheckModelOptions.map((option) => (
                            <SelectItem key={option.value} value={option.value}>
                              {option.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  </FormField>
                ) : (
                  <Display>{localModelData.model_info?.health_check_model || "Not Set"}</Display>
                )}
              </div>
            )}

            {isEditing ? (
              <>
                <FormField
                  control={form.control}
                  name="cache_control"
                  label={
                    <>
                      {CACHE_CONTROL_LABEL}
                      <Hint text={CACHE_CONTROL_TOOLTIP} />
                    </>
                  }
                  orientation="horizontal"
                >
                  {({ id, value, onChange, onBlur }) => (
                    <Switch
                      id={id}
                      onBlur={onBlur}
                      checked={Boolean(value)}
                      onCheckedChange={(checked: boolean) => {
                        onChange(checked);
                        setShowCacheControl(checked);
                      }}
                    />
                  )}
                </FormField>
                {showCacheControl && (
                  <FormField control={form.control} name="cache_control_injection_points">
                    {({ value, onChange }) => (
                      <CacheControlInjectionPoints
                        value={(value as CacheControlInjectionPoint[]) ?? []}
                        onChange={onChange}
                      />
                    )}
                  </FormField>
                )}
              </>
            ) : (
              <div>
                <FieldLabel>Cache Control</FieldLabel>
                <Display>
                  {localModelData.litellm_params?.cache_control_injection_points ? (
                    <div>
                      <p>Enabled</p>
                      <div className="mt-2">
                        {localModelData.litellm_params.cache_control_injection_points.map((point: any, i: number) => (
                          <div key={i} className="mb-1 text-sm text-muted-foreground">
                            Location: {point.location},{point.role && <span> Role: {point.role}</span>}
                            {point.index !== undefined && <span> Index: {point.index}</span>}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    "Disabled"
                  )}
                </Display>
              </div>
            )}

            <div>
              <FieldLabel>Model Info</FieldLabel>
              {isEditing ? (
                <FormField control={form.control} name="model_info">
                  {({ value, ...control }) => (
                    <Textarea
                      {...control}
                      rows={4}
                      placeholder={'{"gpt-4": 100, "claude-v1": 200}'}
                      defaultValue={JSON.stringify(modelData.model_info, null, 2)}
                    />
                  )}
                </FormField>
              ) : (
                <Display>
                  <pre className="mt-1 overflow-auto rounded-sm bg-muted p-2 text-xs">
                    {JSON.stringify(localModelData.model_info, null, 2)}
                  </pre>
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>
                LiteLLM Params
                <DocsHint
                  text="Optional litellm params used for making a litellm.completion() call. Some params are automatically added by LiteLLM."
                  href="https://docs.litellm.ai/docs/completion/input"
                />
              </FieldLabel>
              {isEditing ? (
                <FormField control={form.control} name="litellm_extra_params">
                  {({ value, ...control }) => (
                    <Textarea
                      {...control}
                      value={(value as string) ?? ""}
                      rows={4}
                      placeholder={'{\n  "rpm": 100,\n  "timeout": 0,\n  "stream_timeout": 0\n}'}
                    />
                  )}
                </FormField>
              ) : (
                <Display>
                  <pre className="mt-1 overflow-auto rounded-sm bg-muted p-2 text-xs">
                    {JSON.stringify(localModelData.litellm_params, null, 2)}
                  </pre>
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>Team ID</FieldLabel>
              <Display>{modelData.model_info.team_id || "Not Set"}</Display>
            </div>
          </div>

          {isEditing && (
            <div className="mt-6 flex justify-end gap-2">
              <Button type="submit" variant="secondary" onClick={cancel} disabled={isSaving}>
                Cancel
              </Button>
              <Button type="submit" disabled={isSaving} aria-busy={isSaving}>
                {isSaving && <UiLoadingSpinner className="size-4" />}
                Save Changes
              </Button>
            </div>
          )}
        </div>
      </form>
    </TooltipProvider>
  );
};

export default ModelInfoEditForm;
