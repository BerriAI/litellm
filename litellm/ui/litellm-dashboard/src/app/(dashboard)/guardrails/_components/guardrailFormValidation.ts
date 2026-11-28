import type { TFunction } from "i18next";
import { z } from "zod";
import { readRecord } from "./GuardrailFormField";
import { shouldRenderContentFilterConfigSettings } from "./guardrail_info_helpers";

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

export interface GuardrailProviderParamDescriptor {
  param: string;
  description: string;
  required: boolean;
  default_value?: string;
  options?: string[];
  type?: string;
  fields?: { [key: string]: GuardrailProviderParamDescriptor };
  dict_key_options?: string[];
  dict_value_type?: string;
}

export interface GuardrailProviderParamsMap {
  [provider: string]: { [key: string]: GuardrailProviderParamDescriptor };
}

const CONTENT_FILTER_FIELDS_TO_SKIP = new Set([
  "patterns",
  "blocked_words",
  "blocked_words_file",
  "categories",
  "competitor_intent_config",
]);

export const isBlankGuardrailValue = (value: unknown): boolean => {
  if (value === undefined || value === null || value === "") {
    return true;
  }
  return Array.isArray(value) && value.length === 0;
};

interface ValidateProviderFieldsParams {
  values: Record<string, unknown>;
  selectedProvider: string | null;
  providerFields: Record<string, GuardrailProviderParamDescriptor> | undefined;
  t: TFunction;
  ctx: z.RefinementCtx;
}

export const validateProviderFields = ({
  values,
  selectedProvider,
  providerFields,
  t,
  ctx,
}: ValidateProviderFieldsParams): void => {
  if (!providerFields) return;
  const isContentFilterProvider = shouldRenderContentFilterConfigSettings(selectedProvider);

  const walk = (fields: Record<string, GuardrailProviderParamDescriptor>, parentKey = "") => {
    for (const [fieldKey, field] of Object.entries(fields)) {
      if (fieldKey === "ui_friendly_name") continue;
      if (fieldKey === "optional_params" && field.type === "nested" && field.fields) continue;
      if (isContentFilterProvider && CONTENT_FILTER_FIELDS_TO_SKIP.has(fieldKey)) continue;

      const fullFieldKey = parentKey ? `${parentKey}:${fieldKey}` : fieldKey;
      if (field.type === "nested" && field.fields) {
        walk(field.fields, fullFieldKey);
        continue;
      }

      const value = values[fullFieldKey];
      if (field.type === "object") {
        if (value !== undefined && !isPlainObject(value)) {
          ctx.addIssue({
            code: "custom",
            path: [fullFieldKey],
            message: t("guardrails.providerFields.errors.invalidJsonObject", { field: fieldKey }),
          });
        }
      } else if (field.required && isBlankGuardrailValue(value)) {
        ctx.addIssue({
          code: "custom",
          path: [fullFieldKey],
          message: t("guardrails.providerFields.errors.required", { field: fieldKey }),
        });
      }
    }
  };

  walk(providerFields);
};

export const validateOptionalParams = (
  values: Record<string, unknown>,
  providerFields: Record<string, GuardrailProviderParamDescriptor> | undefined,
  t: TFunction,
  ctx: z.RefinementCtx,
): void => {
  const optionalFields = providerFields?.optional_params?.fields;
  if (!optionalFields) return;
  for (const [fieldKey, field] of Object.entries(optionalFields)) {
    if (field.type === "dict" && field.dict_key_options) continue;
    if (field.required) {
      const value = readRecord(values.optional_params, fieldKey);
      if (isBlankGuardrailValue(value)) {
        ctx.addIssue({
          code: "custom",
          path: ["optional_params", fieldKey],
          message: t("guardrails.optionalParams.errors.required", { field: fieldKey }),
        });
      }
    }
  }
};
