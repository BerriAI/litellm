import type { Resolver, ResolverResult } from "react-hook-form";

import { InputSchema, InputSchemaProperty } from "@/components/mcp_tools/types";

export interface ToolArgumentField {
  readonly key: string;
  readonly prop: InputSchemaProperty;
  readonly required: boolean;
}

export interface ToolArgumentsFormValues {
  args: unknown[];
}

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const isJsonField = (prop: InputSchemaProperty): boolean => prop.type === "object" || prop.type === "array";

export const toolArgumentFields = (schema: InputSchema): readonly ToolArgumentField[] =>
  Object.entries(schema.properties ?? {}).map(([key, prop]) => ({
    key,
    prop,
    required: schema.required?.includes(key) ?? false,
  }));

type ParsedJson = { readonly kind: "ok"; readonly value: unknown } | { readonly kind: "invalid" };

const parseJson = (raw: unknown): ParsedJson => {
  if (typeof raw !== "string") return { kind: "ok", value: raw };
  try {
    return { kind: "ok", value: JSON.parse(raw) };
  } catch {
    return { kind: "invalid" };
  }
};

const isBlank = (value: unknown): boolean => value === undefined || value === null || value === "";

export const validateToolArgument = (field: ToolArgumentField, value: unknown): string | undefined => {
  const normalized = typeof value === "string" ? value.trim() : value;
  if (field.required && isBlank(normalized)) {
    return `Please enter ${field.key}`;
  }
  if (!isJsonField(field.prop) || (isBlank(value) && !field.required)) {
    return undefined;
  }
  const parsed = parseJson(value);
  if (parsed.kind === "invalid") {
    return "Invalid JSON";
  }
  if (field.prop.type === "object" && !isPlainObject(parsed.value)) {
    return "Please enter a JSON object";
  }
  if (field.prop.type === "array" && !Array.isArray(parsed.value)) {
    return "Please enter a JSON array";
  }
  return undefined;
};

const coerceArgument = (prop: InputSchemaProperty, value: unknown): unknown => {
  const normalized = typeof value === "string" ? value.trim() : value;
  switch (prop.type) {
    case "boolean":
      return normalized === "true" || normalized === true;
    case "number":
    case "integer": {
      const numeric = Number(normalized);
      if (Number.isNaN(numeric)) return normalized;
      return prop.type === "integer" ? Math.trunc(numeric) : numeric;
    }
    case "object":
    case "array": {
      const parsed = parseJson(normalized);
      if (parsed.kind === "invalid") return normalized;
      if (prop.type === "object" && isPlainObject(parsed.value)) return parsed.value;
      if (prop.type === "array" && Array.isArray(parsed.value)) return parsed.value;
      return normalized;
    }
    case "string":
      return String(normalized);
    default:
      return normalized;
  }
};

export const buildToolCallArguments = (
  fields: readonly ToolArgumentField[],
  values: readonly unknown[],
): Record<string, unknown> =>
  Object.fromEntries(
    fields
      .map((field, index) => ({ field, value: values[index] }))
      .filter(({ value }) => !isBlank(typeof value === "string" ? value.trim() : value))
      .map(({ field, value }) => [field.key, coerceArgument(field.prop, value)]),
  );

export const toolArgumentsResolver =
  (fields: readonly ToolArgumentField[]): Resolver<ToolArgumentsFormValues> =>
  (values): ResolverResult<ToolArgumentsFormValues> => {
    const issues = fields
      .map((field, index) => ({ index, message: validateToolArgument(field, values.args[index]) }))
      .filter((issue): issue is { index: number; message: string } => issue.message !== undefined);

    if (issues.length === 0) {
      return { values, errors: {} };
    }

    return {
      values: {},
      errors: {
        args: Object.fromEntries(issues.map(({ index, message }) => [index, { type: "validate", message }])),
      },
    };
  };

export const hasNestedParamsSchema = (schema: InputSchema): boolean => {
  const params = schema.properties?.params;
  if (params === undefined) return false;
  return params.type === "object" && params.properties !== undefined;
};

function buildArrayItems(items: InputSchemaProperty | InputSchemaProperty[] | undefined): unknown[] {
  if (!items) return [];
  if (Array.isArray(items)) {
    return items.map((item) => buildDefaultValue(item)).filter((value) => value !== undefined);
  }
  const itemDefault = buildDefaultValue(items);
  return itemDefault === undefined ? [] : [itemDefault];
}

function buildObjectDefault(prop: InputSchemaProperty, effectiveDefault: unknown): Record<string, unknown> {
  const base: Record<string, unknown> = isPlainObject(effectiveDefault) ? effectiveDefault : {};
  if (!prop.properties) return { ...base };
  return {
    ...base,
    ...Object.fromEntries(
      Object.entries(prop.properties).map(([childKey, childProp]) => [
        childKey,
        buildDefaultValue(childProp, base[childKey]),
      ]),
    ),
  };
}

function buildArrayDefault(prop: InputSchemaProperty, effectiveDefault: unknown): unknown {
  if (Array.isArray(effectiveDefault)) {
    const itemSchema = prop.items;
    if (!itemSchema) return effectiveDefault;
    if (effectiveDefault.length === 0) {
      const sample = buildArrayItems(itemSchema);
      return sample.length > 0 ? sample : effectiveDefault;
    }
    if (Array.isArray(itemSchema)) {
      return effectiveDefault.map((value: unknown, index: number) =>
        buildDefaultValue(itemSchema[index] ?? itemSchema[itemSchema.length - 1], value),
      );
    }
    return effectiveDefault.map((value: unknown) => buildDefaultValue(itemSchema, value));
  }
  if (effectiveDefault !== undefined) return effectiveDefault;
  return buildArrayItems(prop.items);
}

function buildDefaultValue(prop: InputSchemaProperty | undefined, overrideDefault?: unknown): unknown {
  if (!prop) return undefined;
  const effectiveDefault: unknown = overrideDefault !== undefined ? overrideDefault : prop.default;

  if (prop.type === "object") return buildObjectDefault(prop, effectiveDefault);
  if (prop.type === "array") return buildArrayDefault(prop, effectiveDefault);
  if (effectiveDefault !== undefined) return effectiveDefault;

  switch (prop.type) {
    case "integer":
    case "number":
      return 0;
    case "boolean":
      return false;
    default:
      return "";
  }
}

export const initialArgumentValues = (fields: readonly ToolArgumentField[]): unknown[] =>
  fields.map(({ prop }) => {
    const defaultValue = buildDefaultValue(prop);
    if (isJsonField(prop)) {
      return JSON.stringify(defaultValue ?? (prop.type === "array" ? [] : {}), null, 2);
    }
    return defaultValue;
  });

export const argumentsFormKey = (schema: InputSchema): string => JSON.stringify(schema);
