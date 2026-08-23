import React, { useState, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Info } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import type { UseFormSetValue } from "react-hook-form";
import { getOpenAPISchema } from "../networking";
import { formatLabel } from "@/utils/textUtils";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { MountedFormField, type MountedFormValues } from "./MountedFormField";

interface SchemaVariant {
  type?: string;
  enum?: string[];
  items?: { type?: string; $ref?: string };
  $ref?: string;
  format?: string;
}

interface SchemaProperty extends SchemaVariant {
  title?: string;
  description?: string;
  anyOf?: SchemaVariant[];
}

interface OpenAPISchema {
  properties: {
    [key: string]: SchemaProperty;
  };
  required?: string[];
}

interface SchemaFormFieldsProps {
  schemaComponent: string;
  excludedFields?: string[];
  setValue: UseFormSetValue<MountedFormValues>;
  overrideLabels?: { [key: string]: string };
  overrideTooltips?: { [key: string]: string };
  customValidation?: {
    [key: string]: (rule: unknown, value: unknown) => Promise<void>;
  };
  defaultValues?: { [key: string]: unknown };
}

// Define which fields should be parsed as JSON
export const jsonFields = [
  "metadata",
  "config",
  "aliases",
  "permissions",
  "model_rpm_limit",
  "model_tpm_limit",
  "mcp_rpm_limit",
  "default_estimated_output_tokens_per_model",
  "allowed_vector_store_indexes",
];

const resolveVariant = (property: SchemaProperty): SchemaVariant => {
  if (property.type || !property.anyOf) return property;
  return (
    property.anyOf.find((variant) => variant.type !== undefined && variant.type !== "null") ??
    property.anyOf.find((variant) => variant.$ref !== undefined) ??
    property
  );
};

// Helper function to determine if a field should be treated as JSON
const isJSONField = (key: string, property: SchemaProperty): boolean => {
  if (jsonFields.includes(key) || property.format === "json") return true;
  const variant = resolveVariant(property);
  if (variant.type === "object" || (variant.type === undefined && variant.$ref !== undefined)) return true;
  return variant.type === "array" && (variant.items?.$ref !== undefined || variant.items?.type === "object");
};

const isStringListField = (key: string, property: SchemaProperty): boolean =>
  !isJSONField(key, property) && resolveVariant(property).type === "array";

// Helper function to validate JSON input
const validateJSON = (value: string): boolean => {
  if (!value) return true;
  try {
    JSON.parse(value);
    return true;
  } catch {
    return false;
  }
};

const isBlank = (value: unknown): boolean => value === undefined || value === null || value === "";

const toSchemaNumber = (raw: string, isInteger: boolean): number | null => {
  if (raw === "") return null;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return null;
  return isInteger ? Math.trunc(parsed) : parsed;
};

const messageOf = (error: unknown): string => (error instanceof Error ? error.message : String(error));

const fieldLabels: { [key: string]: string } = {
  key: "Custom Key",
  budget_id: "Budget ID",
  model_rpm_limit: "Model RPM Limits",
  model_tpm_limit: "Model TPM Limits",
  mcp_rpm_limit: "MCP Server RPM Limits",
};

const fieldTooltips: { [key: string]: string } = {
  key: "Bring your own key value instead of an auto-generated one. Must start with 'sk-' and be at least 16 characters",
  budget_id: "Attach an existing budget (created via /budget/new) to this key",
  soft_budget: "Spend threshold that triggers an alert without blocking the key",
  send_invite_email: "Send an invite email to this key's user",
  max_parallel_requests: "Maximum number of concurrent requests. Requests beyond this limit receive a 429 error",
  allowed_cache_controls: "Cache control values requests with this key may use, e.g. no-cache, no-store",
  config: "Key-specific configuration that overrides values in config.yaml",
  permissions: 'Key-specific permissions, e.g. {"allow_pii_controls": true}',
  model_rpm_limit: "Requests-per-minute limit per model",
  model_tpm_limit: "Tokens-per-minute limit per model",
  mcp_rpm_limit: "Requests-per-minute limit per MCP server, keyed by server name",
  default_estimated_output_tokens:
    "Output tokens reserved for TPM limiting when a request omits max_tokens (proxy admin only)",
  default_estimated_output_tokens_per_model:
    "Per-model override of the default estimated output tokens (proxy admin only)",
  blocked: "Block this key from making any requests",
  enforced_params: "Request parameters every call made with this key must include (Enterprise)",
  allowed_routes: "Proxy routes this key may call. Supports wildcards, e.g. /keys/*",
  allowed_vector_store_indexes: "Vector store indexes this key may access, with per-index permissions",
};

const jsonPlaceholders: { [key: string]: string } = {
  metadata: '{"team": "research"}',
  config: '{"setting": "value"}',
  aliases: '{"my-alias": "gpt-4o"}',
  permissions: '{"allow_pii_controls": true}',
  model_rpm_limit: '{"gpt-4o": 100}',
  model_tpm_limit: '{"gpt-4o": 100000}',
  mcp_rpm_limit: '{"github": 100}',
  default_estimated_output_tokens_per_model: '{"gpt-4o": 4096}',
  allowed_vector_store_indexes: '[{"index_name": "my-index", "index_permissions": ["read"]}]',
};

const textPlaceholders: { [key: string]: string } = {
  key: "sk-...",
};

const getFieldHelp = (key: string, property: SchemaProperty, type: string): string => {
  // Default help text based on type
  const defaultHelp =
    {
      string: "Text input",
      number: "Numeric input",
      integer: "Whole number input",
      boolean: "Toggle on/off",
      array: "Press Enter to add each value",
    }[type] || "Text input";

  // Specific field help text
  const specificHelp: { [key: string]: string } = {
    max_budget: "Enter maximum budget in USD (e.g., 100.50)",
    soft_budget: "Enter alert threshold in USD (e.g., 50)",
    budget_duration: "Select a time period for budget reset",
    budget_id: "Enter the id of an existing budget",
    tpm_limit: "Enter maximum tokens per minute (whole number)",
    rpm_limit: "Enter maximum requests per minute (whole number)",
    max_parallel_requests: "Enter maximum concurrent requests (whole number)",
    duration: "Enter duration (e.g., 30s, 24h, 7d)",
    key: "Must start with 'sk-' and be at least 16 characters",
    metadata: 'Enter JSON object with key-value pairs\nExample: {"team": "research", "project": "nlp"}',
    config: 'Enter configuration as JSON object\nExample: {"setting": "value"}',
    permissions: 'Enter permissions as JSON object\nExample: {"allow_pii_controls": true}',
    enforced_params: "Press Enter to add each required parameter (e.g., user, metadata.generation_name)",
    allowed_cache_controls: "Press Enter to add each cache control value (e.g., no-cache, no-store)",
    allowed_routes: "Press Enter to add each route or wildcard pattern (e.g., /chat/completions, /keys/*)",
    model_rpm_limit: 'Enter JSON mapping model to requests per minute\nExample: {"gpt-4o": 100}',
    model_tpm_limit: 'Enter JSON mapping model to tokens per minute\nExample: {"gpt-4o": 100000}',
    mcp_rpm_limit: 'Enter JSON mapping MCP server to requests per minute\nExample: {"github": 100}',
    default_estimated_output_tokens_per_model: 'Enter JSON mapping model to tokens\nExample: {"gpt-4o": 4096}',
    allowed_vector_store_indexes:
      'Enter JSON array of indexes\nExample: [{"index_name": "my-index", "index_permissions": ["read"]}]',
    aliases: 'Enter aliases as JSON object\nExample: {"alias1": "value1", "alias2": "value2"}',
    models: "Select one or more model names",
    key_alias: "Enter a unique identifier for this key",
    tags: "Enter comma-separated tag strings",
  };

  // Get specific help text or use default based on type
  const helpText = specificHelp[key] || defaultHelp;

  // Add format requirements for special cases
  if (isJSONField(key, property)) {
    return `${helpText}\nMust be valid JSON format`;
  }

  const enumValues = property.enum ?? resolveVariant(property).enum;
  if (enumValues) {
    return `Select from available options\nAllowed values: ${enumValues.join(", ")}`;
  }

  return helpText;
};

const SchemaFormFields: React.FC<SchemaFormFieldsProps> = ({
  schemaComponent,
  excludedFields = [],
  setValue,
  overrideLabels = {},
  overrideTooltips = {},
  customValidation = {},
  defaultValues = {},
}) => {
  const [schemaProperties, setSchemaProperties] = useState<OpenAPISchema | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchOpenAPISchema = async () => {
      try {
        const schema = await getOpenAPISchema();
        const componentSchema = schema.components.schemas[schemaComponent];

        if (!componentSchema) {
          throw new Error(`Schema component "${schemaComponent}" not found`);
        }

        setSchemaProperties(componentSchema);

        Object.keys(componentSchema.properties)
          .filter((key) => !excludedFields.includes(key) && defaultValues[key] !== undefined)
          .forEach((key) => {
            setValue(key, defaultValues[key]);
          });
      } catch (error) {
        console.error("Schema fetch error:", error);
        setError(error instanceof Error ? error.message : "Failed to fetch schema");
      }
    };

    fetchOpenAPISchema();
  }, [schemaComponent, setValue, excludedFields]);

  const getPropertyType = (property: SchemaProperty): string => resolveVariant(property).type ?? "string";

  const renderFormItem = (key: string, property: SchemaProperty) => {
    const type = getPropertyType(property);
    const isRequired = schemaProperties?.required?.includes(key);

    const label = overrideLabels[key] || fieldLabels[key] || property.title || formatLabel(key);
    const tooltip = overrideTooltips[key] || property.description || fieldTooltips[key];

    const validate = {
      ...(isRequired && {
        required: (value: unknown) => (isBlank(value) ? `${label} is required` : true),
      }),
      ...(customValidation[key] && {
        custom: async (value: unknown) => {
          try {
            await customValidation[key](null, value);
            return true;
          } catch (thrown) {
            return messageOf(thrown);
          }
        },
      }),
      ...(isJSONField(key, property) && {
        json: (value: unknown) =>
          value && !validateJSON(value as string) ? "Please enter valid JSON" : (true as const),
      }),
    };

    const formLabel = tooltip ? (
      <span>
        {label}{" "}
        <SimpleTooltip content={tooltip}>
          <Info className="ml-1 inline size-3.5 align-text-bottom" />
        </SimpleTooltip>
      </span>
    ) : (
      label
    );

    return (
      <MountedFormField
        key={key}
        label={formLabel}
        name={key}
        className="mt-8"
        required={isRequired}
        rules={Object.keys(validate).length > 0 ? { validate } : undefined}
        defaultValue={defaultValues[key]}
        help={
          <span className="block text-xs whitespace-pre-line text-muted-foreground">
            {getFieldHelp(key, property, type)}
          </span>
        }
      >
        {(control) => {
          if (isJSONField(key, property)) {
            return (
              <Textarea
                {...control}
                value={(control.value as string | undefined) ?? ""}
                onChange={(event) => control.onChange(event.target.value === "" ? undefined : event.target.value)}
                rows={4}
                placeholder={jsonPlaceholders[key] ?? "Enter as JSON"}
                className="font-mono"
              />
            );
          }
          if (isStringListField(key, property)) {
            return (
              <MultiSelect
                id={control.id}
                options={[]}
                value={(control.value as string[] | undefined) ?? []}
                onValueChange={(next) => control.onChange(next.length > 0 ? next : undefined)}
                placeholder="Type a value and press Enter"
                allowCustomValues
              />
            );
          }
          if (type === "boolean") {
            return (
              <Switch
                id={control.id}
                name={control.name}
                checked={control.value === true}
                onCheckedChange={(checked) => control.onChange(checked)}
                aria-required={control["aria-required"]}
                aria-invalid={control["aria-invalid"]}
                aria-describedby={control["aria-describedby"]}
              />
            );
          }
          const enumValues = property.enum ?? resolveVariant(property).enum;
          if (enumValues) {
            return (
              <Select value={(control.value as string | undefined) ?? null} onValueChange={control.onChange}>
                <SelectTrigger
                  id={control.id}
                  onBlur={control.onBlur}
                  aria-invalid={control["aria-invalid"]}
                  className="w-full"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {enumValues.map((value) => (
                    <SelectItem key={value} value={value}>
                      {value}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            );
          }
          if (type === "number" || type === "integer") {
            return (
              <Input
                {...control}
                type="number"
                step={type === "integer" ? 1 : "any"}
                value={(control.value as number | undefined) ?? ""}
                onChange={(event) => control.onChange(toSchemaNumber(event.target.value, type === "integer"))}
                className="w-full"
              />
            );
          }
          if (key === "duration") {
            return (
              <Input {...control} value={(control.value as string | undefined) ?? ""} placeholder="eg: 30s, 30h, 30d" />
            );
          }
          return (
            <Input
              {...control}
              value={(control.value as string | undefined) ?? ""}
              placeholder={textPlaceholders[key] ?? ""}
            />
          );
        }}
      </MountedFormField>
    );
  };

  if (error) {
    return <div className="text-destructive">Error: {error}</div>;
  }

  if (!schemaProperties?.properties) {
    return null;
  }

  return (
    <div>
      {Object.entries(schemaProperties.properties)
        .filter(([key]) => !excludedFields.includes(key))
        .map(([key, property]) => renderFormItem(key, property))}
    </div>
  );
};

export default SchemaFormFields;
