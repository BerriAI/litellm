import React, { useState, useEffect } from "react";
import {
  guardrail_provider_map,
  populateGuardrailProviders,
  populateGuardrailProviderMap,
  shouldRenderContentFilterConfigSettings,
} from "./guardrail_info_helpers";
import { getGuardrailProviderSpecificParams } from "@/components/networking";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { PasswordInput } from "@/components/shared/PasswordInput";
import NumericalInput from "@/components/shared/numerical_input";
import { FieldGroup } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import {
  asStringArray,
  asText,
  GuardrailField,
  labelWithHint,
  readRecord,
  requiredRule,
  type GuardrailFieldControlProps,
  type GuardrailFormControl,
} from "./GuardrailFormField";

interface GuardrailProviderFieldsProps {
  selectedProvider: string | null;
  control: GuardrailFormControl;
  accessToken?: string | null;
  providerParams?: ProviderParamsResponse | null;
  value?: Record<string, unknown> | null;
}

interface ProviderParam {
  param: string;
  description: string;
  required: boolean;
  default_value?: string | number;
  options?: string[];
  type?: string;
  fields?: { [key: string]: ProviderParam };
  dict_key_options?: string[];
  dict_value_type?: string;
  min?: number;
  max?: number;
  step?: number;
}

interface ProviderParamsResponse {
  [provider: string]: { [key: string]: ProviderParam };
}

const BOOLEAN_ITEMS = [
  { label: "True", value: true },
  { label: "False", value: false },
];

const isSecretKey = (fieldKey: string): boolean =>
  fieldKey.includes("password") || fieldKey.includes("secret") || fieldKey.includes("key");

interface ProviderFieldInputProps {
  descriptor: ProviderParam;
  fieldKey: string;
  control: GuardrailFieldControlProps;
}

const ProviderFieldInput: React.FC<ProviderFieldInputProps> = ({ descriptor, fieldKey, control }) => {
  const { id, value, onChange, onBlur, ref, name, ...aria } = control;

  if (descriptor.type === "select" && descriptor.options) {
    return (
      <Select
        items={descriptor.options.map((option) => ({ label: option, value: option }))}
        value={asText(value) || null}
        onValueChange={(next: string | null) => onChange(next)}
      >
        <SelectTrigger id={id} className="w-full" {...aria}>
          <SelectValue placeholder={descriptor.description} />
        </SelectTrigger>
        <SelectContent>
          {descriptor.options.map((option) => (
            <SelectItem key={option} value={option}>
              {option}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    );
  }

  if (descriptor.type === "multiselect" && descriptor.options) {
    return (
      <MultiSelect
        id={id}
        options={descriptor.options.map((option) => ({ label: option, value: option }))}
        value={asStringArray(value)}
        onValueChange={onChange}
        placeholder={descriptor.description}
      />
    );
  }

  if (descriptor.type === "bool" || descriptor.type === "boolean") {
    return (
      <Select
        items={BOOLEAN_ITEMS}
        value={typeof value === "boolean" ? value : null}
        onValueChange={(next: boolean | null) => onChange(next)}
      >
        <SelectTrigger id={id} className="w-full" {...aria}>
          <SelectValue placeholder={descriptor.description} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={true}>True</SelectItem>
          <SelectItem value={false}>False</SelectItem>
        </SelectContent>
      </Select>
    );
  }

  if (descriptor.type === "percentage" && descriptor.min != null && descriptor.max != null) {
    return (
      <div className="w-full">
        <Slider
          id={id}
          min={descriptor.min}
          max={descriptor.max}
          step={descriptor.step ?? 0.1}
          value={typeof value === "number" ? value : descriptor.min}
          onValueChange={(next: number | readonly number[]) => onChange(Array.isArray(next) ? next[0] : next)}
          onBlur={onBlur}
        />
        <div className="mt-1 flex justify-between text-xs text-muted-foreground">
          <span>0%</span>
          <span>50%</span>
          <span>100%</span>
        </div>
      </div>
    );
  }

  if (descriptor.type === "number") {
    return (
      <NumericalInput
        id={id}
        name={name}
        step={1}
        placeholder={descriptor.description}
        value={asText(value)}
        onChange={onChange}
        onBlur={onBlur}
        {...aria}
      />
    );
  }

  if (isSecretKey(fieldKey)) {
    return (
      <PasswordInput
        id={id}
        name={name}
        ref={ref}
        placeholder={descriptor.description}
        value={asText(value)}
        onChange={onChange}
        onBlur={onBlur}
        {...aria}
      />
    );
  }

  return (
    <Input
      id={id}
      name={name}
      ref={ref}
      placeholder={descriptor.description}
      value={asText(value)}
      onChange={onChange}
      onBlur={onBlur}
      {...aria}
    />
  );
};

const GuardrailProviderFields: React.FC<GuardrailProviderFieldsProps> = ({
  selectedProvider,
  control,
  accessToken,
  providerParams: providerParamsProp = null,
  value = null,
}) => {
  const [loading, setLoading] = useState(false);
  const [providerParams, setProviderParams] = useState<ProviderParamsResponse | null>(providerParamsProp);
  const [error, setError] = useState<string | null>(null);

  // Fetch provider-specific parameters when component mounts
  useEffect(() => {
    if (providerParamsProp) {
      // Props updated externally
      setProviderParams(providerParamsProp);
      return;
    }

    const fetchProviderParams = async () => {
      if (!accessToken) return;

      setLoading(true);
      setError(null);

      try {
        const data = await getGuardrailProviderSpecificParams(accessToken);
        setProviderParams(data);

        // Populate dynamic providers from API response
        populateGuardrailProviders(data);
        populateGuardrailProviderMap(data);
      } catch (error) {
        console.error("Error fetching provider params:", error);
        setError("Failed to load provider parameters");
      } finally {
        setLoading(false);
      }
    };

    // Only fetch if not provided via props
    if (!providerParamsProp) {
      fetchProviderParams();
    }
  }, [accessToken, providerParamsProp]);

  // If no provider is selected, don't render anything
  if (!selectedProvider) {
    return null;
  }

  // Show loading state
  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <UiLoadingSpinner className="size-4" />
        Loading provider parameters...
      </div>
    );
  }

  // Show error state
  if (error) {
    return <div className="text-destructive">{error}</div>;
  }

  // Get the provider key matching the selected provider in the guardrail_provider_map
  const providerKey = guardrail_provider_map[selectedProvider]?.toLowerCase();

  // Get parameters for the selected provider
  const providerFields = providerParams && providerParams[providerKey];

  if (!providerFields || Object.keys(providerFields).length === 0) {
    return <div>No configuration fields available for this provider.</div>;
  }

  // Fields to skip for content filter provider (handled in dedicated steps)
  const contentFilterFieldsToSkip = new Set([
    "patterns",
    "blocked_words",
    "blocked_words_file",
    "categories",
    "severity_threshold",
    "pattern_redaction_format",
    "keyword_redaction_tag",
  ]);

  const isContentFilterProvider = shouldRenderContentFilterConfigSettings(selectedProvider);

  // Convert object to array of entries and render fields
  const renderFields = (fields: { [key: string]: ProviderParam }, parentKey = "", parentValue?: unknown) => {
    return Object.entries(fields).map(([fieldKey, field]) => {
      // ":" keeps nested children out of the submitted object graph: nothing binds the parent
      const fullFieldKey = parentKey ? `${parentKey}:${fieldKey}` : fieldKey;
      const fieldValue = parentValue ? readRecord(parentValue, fieldKey) : value?.[fieldKey];
      // Skip ui_friendly_name - it's metadata for the UI dropdown, not a user configuration field
      if (fieldKey === "ui_friendly_name") {
        return null;
      }

      // Skip optional_params - they are handled in a separate step
      if (fieldKey === "optional_params" && field.type === "nested" && field.fields) {
        return null;
      }

      // Skip content filter specific fields when it's a content filter provider (handled in dedicated steps)
      if (isContentFilterProvider && contentFilterFieldsToSkip.has(fieldKey)) {
        return null;
      }

      // Handle other nested fields (like azure/text_moderations optional_params)
      if (field.type === "nested" && field.fields) {
        return (
          <div key={fullFieldKey}>
            <div className="mb-2 font-medium">{fieldKey}</div>
            <FieldGroup className="ml-4 border-l-2 border-border pl-4">
              {renderFields(field.fields, fullFieldKey, fieldValue)}
            </FieldGroup>
          </div>
        );
      }

      const resolvedInitialValue =
        fieldValue !== undefined ? fieldValue : field.default_value ?? (field.type === "percentage" ? 0.5 : undefined);

      return (
        <GuardrailField
          key={fullFieldKey}
          control={control}
          name={fullFieldKey}
          label={labelWithHint(fieldKey, field.description)}
          rules={field.required ? requiredRule(`${fieldKey} is required`) : undefined}
          defaultValue={resolvedInitialValue}
        >
          {(fieldControl) => <ProviderFieldInput descriptor={field} fieldKey={fieldKey} control={fieldControl} />}
        </GuardrailField>
      );
    });
  };

  return <FieldGroup>{renderFields(providerFields)}</FieldGroup>;
};

export default GuardrailProviderFields;
