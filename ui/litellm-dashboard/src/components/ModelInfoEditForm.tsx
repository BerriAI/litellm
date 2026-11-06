"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { CircleHelp } from "lucide-react";
import type { Dayjs } from "dayjs";
import type { ParseKeys, TFunction } from "i18next";
import * as React from "react";
import { useForm, type Resolver } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { z } from "zod";

import { TagsInput } from "@/app/(dashboard)/guardrails/_components/content_filter/TagsInput";
import { FormField } from "@/components/shared/form/FormField";
import { UtcDateTimeInput } from "@/components/shared/form/UtcDateTimeInput";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { credentialOptions } from "@/components/shared/credentialOptions";
import { AttachedCredential, CredentialNameHint } from "@/components/shared/CredentialIdentity";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import i18n from "@/i18n/i18n";

import CacheControlInjectionPoints, {
  CACHE_CONTROL_LABEL,
  CACHE_CONTROL_TOOLTIP,
  type CacheControlInjectionPoint,
} from "./add_model/cache_control_settings";
import type { Team } from "./key_team_helpers/key_list";
import type { CredentialItem } from "./networking";
import NumericalInput from "./shared/numerical_input";
import type { Tag } from "./tag_management/types";
import { ModelTeamSelect } from "./view_model/ModelTeamSelect";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import { formatPtuUtcDisplay, utcIsoToPickerValue } from "../utils/ptuDatetime";
import { editableModelExtraParams, parseModelExtraParams } from "../utils/modelEditParams";
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
  label: ParseKeys;
  input: "number" | "datetime";
  placeholder?: ParseKeys;
  isCount?: boolean;
}

const PTU_EDIT_FIELDS: PtuEditField[] = [
  {
    name: PTU_COUNT_FIELD,
    label: "modelInfoEditForm.fields.ptuCount.label",
    input: "number",
    placeholder: "modelInfoEditForm.fields.ptuCount.placeholder",
    isCount: true,
  },
  {
    name: PTU_RATE_FIELD,
    label: "modelInfoEditForm.fields.ptuRate.label",
    input: "number",
    placeholder: "modelInfoEditForm.fields.ptuRate.placeholder",
  },
  { name: PTU_START_FIELD, label: "modelInfoEditForm.fields.ptuEffectiveFrom.label", input: "datetime" },
  { name: PTU_END_FIELD, label: "modelInfoEditForm.fields.ptuEffectiveTo.label", input: "datetime" },
];

const PRICING_FIELDS = [
  "input_cost",
  "output_cost",
  "cache_read_cost",
  "cache_write_cost",
  "cache_write_cost_1h",
] as const;
export type TouchedPricingField = (typeof PRICING_FIELDS)[number];

const COST_SOURCES: Record<TouchedPricingField, string> = {
  input_cost: "input_cost_per_token",
  output_cost: "output_cost_per_token",
  cache_read_cost: "cache_read_input_token_cost",
  cache_write_cost: "cache_creation_input_token_cost",
  cache_write_cost_1h: "cache_creation_input_token_cost_above_1hr",
};

const scalar = z.union([z.string(), z.number(), z.null()]).optional();
const textish = z.string().optional();

const isBlank = (value: string | number | null | undefined): boolean => value == null || String(value).trim() === "";

export const perMillionToPerToken = scalar
  .refine((perMillion) => isBlank(perMillion) || (Number.isFinite(Number(perMillion)) && Number(perMillion) >= 0), {
    error: () => i18n.t("modelInfoEditForm.validation.priceNonNegative"),
  })
  .transform((perMillion) => (isBlank(perMillion) ? null : Number(perMillion) / 1_000_000));

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
  input_cost: perMillionToPerToken,
  output_cost: perMillionToPerToken,
  cache_read_cost: perMillionToPerToken,
  cache_write_cost: perMillionToPerToken,
  cache_write_cost_1h: perMillionToPerToken,
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
  litellm_credential_name: z.string().nullish(),
  litellm_extra_params: textish,
  model_info: textish,
  team_id: textish,
};

type ModelEditFieldName = keyof typeof modelEditShape;

const buildSchema = (
  ptuEnabled: boolean,
  isFieldTouched: (field: TouchedPricingField) => boolean,
  t: TFunction = i18n.t.bind(i18n),
) =>
  z.object(modelEditShape).superRefine((values, ctx) => {
    const reject = (path: ModelEditFieldName, message: string) =>
      ctx.addIssue({ code: "custom", path: [path], message });

    if (parseModelExtraParams(values.litellm_extra_params) === null) {
      reject("litellm_extra_params", t("modelInfoEditForm.validation.invalidJson"));
    }

    // antd validates only mounted fields, and the PTU block does not render when the flag is off.
    if (!ptuEnabled) {
      return;
    }

    if (!isPositiveWholePtuCount(values.ptu_count)) {
      reject("ptu_count", t("modelInfoEditForm.validation.ptuCountRange", { max: MAX_PTU_COUNT.toLocaleString() }));
    }
    if (!isNonNegativePtuRate(values.cost_per_ptu_per_hour)) {
      reject(
        "cost_per_ptu_per_hour",
        t("modelInfoEditForm.validation.ptuRateRange", { max: MAX_COST_PER_PTU_PER_HOUR.toLocaleString() }),
      );
    }
    if (isFilledPtuValue(values.ptu_count) !== isFilledPtuValue(values.cost_per_ptu_per_hour)) {
      const message = t("modelInfoEditForm.validation.ptuPair");
      reject("ptu_count", message);
      reject("cost_per_ptu_per_hour", message);
    }
    if (isFilledPtuValue(values.ptu_count) && !isFilledPtuValue(values.ptu_effective_from)) {
      reject("ptu_effective_from", t("modelInfoEditForm.validation.ptuStartRequired"));
    }
    if (!ptuWindowIsOrdered(values.ptu_effective_from, values.ptu_effective_to)) {
      const message = t("modelInfoEditForm.validation.ptuWindowOrder");
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
        reject(field, t("modelInfoEditForm.validation.ptuPricing"));
      }
    }
  });

type ModelEditSchema = ReturnType<typeof buildSchema>;
export type ModelEditFormValues = z.input<ModelEditSchema>;
export type ModelEditSubmitValues = z.output<ModelEditSchema>;

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
  cache_write_cost_1h: perMillionTokens(
    localModelData.litellm_params?.cache_creation_input_token_cost_above_1hr,
    localModelData.model_info?.cache_creation_input_token_cost_above_1hr,
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
  litellm_credential_name: localModelData.litellm_params?.litellm_credential_name ?? null,
  litellm_extra_params: JSON.stringify(editableModelExtraParams(localModelData.litellm_params ?? {}), null, 2),
  team_id: localModelData.model_info?.team_id ?? undefined,
});

const displayCost = (localModelData: any, field: TouchedPricingField, t: TFunction = i18n.t.bind(i18n)): string => {
  const key = COST_SOURCES[field];
  const rate = localModelData?.litellm_params?.[key] ?? localModelData?.model_info?.[key];
  return rate != null ? (Number(rate) * 1_000_000).toFixed(4) : t("modelInfoEditForm.notSet");
};

interface ModelInfoEditFormProps {
  localModelData: any;
  modelData: { model_info: { team_id?: string | null } };
  teamAlias: string | null;
  accessToken: string | null;
  isEditing: boolean;
  isSaving: boolean;
  isWildcardModel: boolean;
  ptuCostAttributionEnabled: boolean;
  showCacheControl: boolean;
  setShowCacheControl: (checked: boolean) => void;
  onCancel: () => void;
  onSubmit: (values: ModelEditSubmitValues, isFieldTouched: (field: TouchedPricingField) => boolean) => Promise<void>;
  modelAccessGroups: string[] | null;
  guardrailsList: string[];
  tagsList: Record<string, Tag>;
  credentialsList: CredentialItem[];
  healthCheckModelOptions: { value: string; label: string }[];
  teams: Team[] | null;
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
  const { t } = useTranslation();
  if (!values) {
    return <>{t("modelInfoEditForm.notSet")}</>;
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
  teamAlias,
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
  teams,
}) => {
  const { t } = useTranslation();
  // Neither RHF's blur-based touchedFields nor its resettable dirtyFields matches antd's touched-on-change.
  const touchedRef = React.useRef<ReadonlySet<string>>(new Set<string>());
  const isFieldTouched = React.useCallback((field: TouchedPricingField) => touchedRef.current.has(field), []);
  const markTouched = (field: string) => {
    touchedRef.current = new Set([...touchedRef.current, field]);
  };

  // react-hook-form refreshes control._options every render, so this rebuild is what the next submit runs.
  const resolver: Resolver<ModelEditFormValues, unknown, ModelEditSubmitValues> = (values, context, options) =>
    zodResolver(buildSchema(ptuCostAttributionEnabled, isFieldTouched, t))(values, context, options);

  const form = useForm<ModelEditFormValues, unknown, ModelEditSubmitValues>({
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
        <Display>{(stored as string) || t("modelInfoEditForm.notSet")}</Display>
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
        <Display>{(stored as string) || t("modelInfoEditForm.notSet")}</Display>
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
        <Display>{displayCost(localModelData, name, t)}</Display>
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
            {textField(
              "model_name",
              t("modelInfoEditForm.fields.modelName.label"),
              t("modelInfoEditForm.fields.modelName.placeholder"),
              localModelData.model_name,
            )}
            {textField(
              "litellm_model_name",
              t("modelInfoEditForm.fields.litellmModelName.label"),
              t("modelInfoEditForm.fields.litellmModelName.placeholder"),
              localModelData.litellm_model_name,
            )}

            {pricingField(
              "input_cost",
              t("modelInfoEditForm.fields.inputCost.label"),
              t("modelInfoEditForm.fields.inputCost.placeholder"),
            )}
            {pricingField(
              "output_cost",
              t("modelInfoEditForm.fields.outputCost.label"),
              t("modelInfoEditForm.fields.outputCost.placeholder"),
            )}

            {ptuCostAttributionEnabled &&
              PTU_EDIT_FIELDS.map((ptuField) => (
                <div key={ptuField.name}>
                  <FieldLabel htmlFor={ptuField.name}>{t(ptuField.label)}</FieldLabel>
                  {isEditing ? (
                    <FormField control={form.control} name={ptuField.name as ModelEditFieldName}>
                      {({ value, onChange, ...control }) =>
                        ptuField.input === "number" ? (
                          <NumericalInput
                            {...control}
                            id={ptuField.name}
                            onChange={onChange}
                            value={value ?? ""}
                            placeholder={ptuField.placeholder && t(ptuField.placeholder)}
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
                        : localModelData?.model_info?.[ptuField.name]) ?? t("modelInfoEditForm.notSet")}
                    </Display>
                  )}
                </div>
              ))}

            {pricingField(
              "cache_read_cost",
              t("modelInfoEditForm.fields.cacheReadCost.label"),
              t("modelInfoEditForm.fields.cacheReadCost.placeholder"),
              t("modelInfoEditForm.fields.cacheReadCost.description"),
            )}
            {pricingField(
              "cache_write_cost",
              t("modelInfoEditForm.fields.cacheWriteCost.label"),
              t("modelInfoEditForm.fields.cacheWriteCost.placeholder"),
              t("modelInfoEditForm.fields.cacheWriteCost.description"),
            )}
            {pricingField(
              "cache_write_cost_1h",
              t("modelInfoEditForm.fields.cacheWriteCost1h.label"),
              t("modelInfoEditForm.fields.cacheWriteCost1h.placeholder"),
              t("modelInfoEditForm.fields.cacheWriteCost1h.description"),
            )}

            {textField(
              "api_base",
              t("modelInfoEditForm.fields.apiBase.label"),
              t("modelInfoEditForm.fields.apiBase.placeholder"),
              localModelData.litellm_params?.api_base,
            )}
            {textField(
              "custom_llm_provider",
              t("modelInfoEditForm.fields.customLlmProvider.label"),
              t("modelInfoEditForm.fields.customLlmProvider.placeholder"),
              localModelData.litellm_params?.custom_llm_provider,
            )}
            {textField(
              "organization",
              t("modelInfoEditForm.fields.organization.label"),
              t("modelInfoEditForm.fields.organization.placeholder"),
              localModelData.litellm_params?.organization,
            )}

            {numberField(
              "tpm",
              t("modelInfoEditForm.fields.tpm.label"),
              t("modelInfoEditForm.fields.tpm.placeholder"),
              localModelData.litellm_params?.tpm,
            )}
            {numberField(
              "rpm",
              t("modelInfoEditForm.fields.rpm.label"),
              t("modelInfoEditForm.fields.rpm.placeholder"),
              localModelData.litellm_params?.rpm,
            )}
            {numberField(
              "max_retries",
              t("modelInfoEditForm.fields.maxRetries.label"),
              t("modelInfoEditForm.fields.maxRetries.placeholder"),
              localModelData.litellm_params?.max_retries,
            )}
            {numberField(
              "timeout",
              t("modelInfoEditForm.fields.timeout.label"),
              t("modelInfoEditForm.fields.timeout.placeholder"),
              localModelData.litellm_params?.timeout,
            )}
            {numberField(
              "stream_timeout",
              t("modelInfoEditForm.fields.streamTimeout.label"),
              t("modelInfoEditForm.fields.streamTimeout.placeholder"),
              localModelData.litellm_params?.stream_timeout,
            )}

            <div>
              <FieldLabel>{t("modelInfoEditForm.fields.modelAccessGroups.label")}</FieldLabel>
              {isEditing ? (
                tagsField(
                  "model_access_group",
                  (modelAccessGroups ?? []).map((group) => ({ value: group, label: group })),
                  t("modelInfoEditForm.fields.modelAccessGroups.placeholder"),
                )
              ) : (
                <Display>
                  <ChipList
                    values={localModelData.model_info?.access_groups}
                    emptyLabel={t("modelInfoEditForm.fields.modelAccessGroups.empty")}
                  />
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>
                {t("modelInfoEditForm.fields.guardrails.label")}
                <DocsHint
                  text={t("modelInfoEditForm.fields.guardrails.hint")}
                  href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                />
              </FieldLabel>
              {isEditing ? (
                tagsField(
                  "guardrails",
                  guardrailsList.map((name) => ({ value: name, label: name })),
                  t("modelInfoEditForm.fields.guardrails.placeholder"),
                )
              ) : (
                <Display>
                  <ChipList
                    values={localModelData.litellm_params?.guardrails}
                    emptyLabel={t("modelInfoEditForm.fields.guardrails.empty")}
                  />
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>
                {t("modelInfoEditForm.fields.knowledgeBases.label")}
                <DocsHint
                  text={t("modelInfoEditForm.fields.knowledgeBases.hint")}
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
                      placeholder={t("modelInfoEditForm.fields.knowledgeBases.placeholder")}
                    />
                  )}
                </FormField>
              ) : (
                <Display>
                  <ChipList
                    values={localModelData.litellm_params?.vector_store_ids}
                    emptyLabel={t("modelInfoEditForm.fields.knowledgeBases.empty")}
                  />
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>{t("modelInfoEditForm.fields.tags.label")}</FieldLabel>
              {isEditing ? (
                tagsField(
                  "tags",
                  Object.values(tagsList).map((tag: Tag) => ({ value: tag.name, label: tag.name })),
                  t("modelInfoEditForm.fields.tags.placeholder"),
                )
              ) : (
                <Display>
                  <ChipList
                    values={localModelData.litellm_params?.tags}
                    emptyLabel={t("modelInfoEditForm.fields.tags.empty")}
                  />
                </Display>
              )}
            </div>

            <div>
              <FieldLabel>{t("modelInfoEditForm.fields.existingCredentials.label")}</FieldLabel>
              {isEditing ? (
                <FormField control={form.control} name="litellm_credential_name">
                  {({ id, value, onChange }) => (
                    <>
                      <SearchSelect
                        inputId={id}
                        placeholder={t("modelInfoEditForm.fields.existingCredentials.placeholder")}
                        options={credentialOptions(credentialsList)}
                        value={(value as string | null) ?? ""}
                        allowClear={false}
                        onValueChange={(selected) => {
                          if (selected !== null) onChange(selected === "" ? null : selected);
                        }}
                      />
                      <CredentialNameHint
                        credentialName={value as string | null | undefined}
                        credentials={credentialsList}
                      />
                    </>
                  )}
                </FormField>
              ) : (
                <Display>
                  <AttachedCredential
                    credentialName={localModelData.litellm_params?.litellm_credential_name}
                    credentials={credentialsList}
                  />
                </Display>
              )}
            </div>

            {isWildcardModel && (
              <div>
                <FieldLabel>{t("modelInfoEditForm.fields.healthCheckModel.label")}</FieldLabel>
                {isEditing ? (
                  <FormField control={form.control} name="health_check_model">
                    {({ id, value, onChange, onBlur }) => (
                      <Select
                        items={healthCheckModelOptions}
                        value={(value as string | null) ?? null}
                        onValueChange={onChange}
                      >
                        <SelectTrigger id={id} className="w-full" onBlur={onBlur}>
                          <SelectValue placeholder={t("modelInfoEditForm.fields.healthCheckModel.placeholder")} />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value={null}>{t("modelInfoEditForm.fields.healthCheckModel.none")}</SelectItem>
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
                  <Display>{localModelData.model_info?.health_check_model || t("modelInfoEditForm.notSet")}</Display>
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
                      {t(CACHE_CONTROL_LABEL)}
                      <Hint text={t(CACHE_CONTROL_TOOLTIP)} />
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
                <FieldLabel>{t("modelInfoEditForm.fields.cacheControl.label")}</FieldLabel>
                <Display>
                  {localModelData.litellm_params?.cache_control_injection_points ? (
                    <div>
                      <p>{t("modelInfoEditForm.fields.cacheControl.enabled")}</p>
                      <div className="mt-2">
                        {localModelData.litellm_params.cache_control_injection_points.map((point: any, i: number) => (
                          <div key={i} className="mb-1 text-sm text-muted-foreground">
                            {t("modelInfoEditForm.fields.cacheControl.location", { location: point.location })}
                            {point.role && (
                              <span> {t("modelInfoEditForm.fields.cacheControl.role", { role: point.role })}</span>
                            )}
                            {point.index !== undefined && (
                              <span> {t("modelInfoEditForm.fields.cacheControl.index", { index: point.index })}</span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    t("modelInfoEditForm.fields.cacheControl.disabled")
                  )}
                </Display>
              </div>
            )}

            <div>
              <FieldLabel>{t("modelInfoEditForm.fields.modelInfo.label")}</FieldLabel>
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
                {t("modelInfoEditForm.fields.litellmParams.label")}
                <DocsHint
                  text={t("modelInfoEditForm.fields.litellmParams.hint")}
                  href="https://docs.litellm.ai/docs/completion/input"
                />
              </FieldLabel>
              {isEditing ? (
                <FormField control={form.control} name="litellm_extra_params">
                  {({ value, ...control }) => (
                    <Textarea
                      {...control}
                      aria-label={t("modelInfoEditForm.fields.litellmParams.label")}
                      value={(value as string) ?? ""}
                      rows={4}
                      placeholder={'{\n  "drop_params": true\n}'}
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
              <FieldLabel>{t("modelInfoEditForm.fields.team.label")}</FieldLabel>
              {isEditing ? (
                <FormField control={form.control} name="team_id">
                  {({ id, value, onChange, onBlur }) => (
                    <ModelTeamSelect id={id} value={value} onChange={onChange} onBlur={onBlur} teams={teams} />
                  )}
                </FormField>
              ) : (
                <Display>
                  {teamAlias
                    ? `${teamAlias} (${localModelData.model_info?.team_id})`
                    : localModelData.model_info?.team_id || t("modelInfoEditForm.notSet")}
                </Display>
              )}
            </div>
          </div>

          {isEditing && (
            <div className="mt-6 flex justify-end gap-2">
              <Button type="submit" variant="secondary" onClick={cancel} disabled={isSaving}>
                {t("modelInfoEditForm.actions.cancel")}
              </Button>
              <Button type="submit" disabled={isSaving} aria-busy={isSaving}>
                {isSaving && <UiLoadingSpinner className="size-4" />}
                {t("modelInfoEditForm.actions.saveChanges")}
              </Button>
            </div>
          )}
        </div>
      </form>
    </TooltipProvider>
  );
};

export default ModelInfoEditForm;
