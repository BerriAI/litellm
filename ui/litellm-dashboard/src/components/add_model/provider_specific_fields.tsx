import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Upload as UploadIcon } from "lucide-react";
import React from "react";
import { useFormContext } from "react-hook-form";
import { requiredRule } from "../common_components/formRules";
import {
  MountedFormField,
  type MountedFieldControlProps,
  type MountedFormValues,
} from "../common_components/MountedFormField";
import { CredentialItem, ProviderCredentialFieldMetadata } from "../networking";
import { provider_map, Providers } from "../provider_info_helpers";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";

interface ProviderSpecificFieldsProps {
  selectedProvider: Providers;
}

const readTextFile = (file: File, onLoaded: (contents: string) => void) => {
  const reader = new FileReader();
  reader.onload = (event) => {
    if (event.target) {
      onLoaded(event.target.result as string);
    }
  };
  reader.readAsText(file);
};

interface ProviderCredentialField {
  key: string;
  label: string;
  placeholder?: string;
  tooltip?: string;
  required?: boolean;
  type?: "text" | "password" | "select" | "upload" | "textarea";
  options?: string[];
  defaultValue?: string;
}

export interface CredentialValues {
  key: string;
  value: string;
}

const getApiVersionFromApiBase = (apiBase: string): string | null => {
  const queryStartIndex = apiBase.indexOf("?");
  if (queryStartIndex === -1) {
    return null;
  }

  const queryString = apiBase.slice(queryStartIndex + 1).split("#")[0];
  const searchParams = new URLSearchParams(queryString);

  return searchParams.get("api_version") || searchParams.get("api-version");
};

const mapFieldMetadataToUiField = (field: ProviderCredentialFieldMetadata): ProviderCredentialField => {
  const type: ProviderCredentialField["type"] =
    field.field_type === "password"
      ? "password"
      : field.field_type === "select"
        ? "select"
        : field.field_type === "upload"
          ? "upload"
          : field.field_type === "textarea"
            ? "textarea"
            : "text";

  return {
    key: field.key,
    label: field.label,
    placeholder: field.placeholder ?? undefined,
    tooltip: field.tooltip ?? undefined,
    required: field.required ?? false,
    type,
    options: field.options ?? undefined,
    defaultValue: field.default_value ?? undefined,
  };
};

// In-memory cache of provider credential fields keyed by provider display name.
// This lets us reuse the data across multiple mounts and also supports
// non-React helpers like createCredentialFromModel.
const providerFieldsByDisplayName: Record<string, ProviderCredentialField[]> = {};

export const createCredentialFromModel = (provider: string, modelData: any): CredentialItem => {
  const enumKey = Object.keys(provider_map).find((key) => provider_map[key].toLowerCase() === provider.toLowerCase());
  if (!enumKey) {
    throw new Error(`Provider ${provider} not found in provider_map`);
  }
  const providerDisplayName = Providers[enumKey as keyof typeof Providers];
  const providerFields = providerFieldsByDisplayName[providerDisplayName] || [];
  const credentialValues: object = {};

  // Go through each field defined for this provider
  providerFields.forEach((field) => {
    const value = modelData.litellm_params[field.key];
    if (value !== undefined) {
      (credentialValues as Record<string, string>)[field.key] = value.toString();
    }
  });

  const credential: CredentialItem = {
    credential_name: `${provider}-credential-${Math.floor(Math.random() * 1000000)}`,
    credential_values: credentialValues,
    credential_info: {
      custom_llm_provider: provider,
      description: `Credential for ${provider}. Created from model ${modelData.model_name}`,
    },
  };

  return credential;
};

const ProviderSpecificFields: React.FC<ProviderSpecificFieldsProps> = ({ selectedProvider }) => {
  const selectedProviderEnum = Providers[selectedProvider as keyof typeof Providers] as Providers;
  const form = useFormContext<MountedFormValues>();
  const credentialsFileRef = React.useRef<HTMLInputElement>(null);
  const pickCredentialsFile =
    (onLoaded: (contents: string) => void) => (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (file?.type === "application/json") {
        readTextFile(file, onLoaded);
      }
    };

  const { data: providerMetadata, isLoading, error: loadError } = useProviderFields();

  // Memoize the expensive cache computation
  const cacheEntries = React.useMemo(() => {
    if (!providerMetadata) {
      return null;
    }

    // Compute cache entries keyed by provider display name and identifiers
    const entries: Record<string, ProviderCredentialField[]> = {};
    providerMetadata.forEach((providerInfo) => {
      const displayName = providerInfo.provider_display_name;
      const mappedFields = providerInfo.credential_fields.map(mapFieldMetadataToUiField);

      // Primary key: human-readable display name
      entries[displayName] = mappedFields;

      // Also cache by backend identifiers so lookups by provider slug work
      if (providerInfo.provider) {
        entries[providerInfo.provider] = mappedFields;
      }
      if (providerInfo.litellm_provider) {
        entries[providerInfo.litellm_provider] = mappedFields;
      }
    });
    return entries;
  }, [providerMetadata]);

  // Sync memoized cache entries to module-level cache
  React.useEffect(() => {
    if (!cacheEntries) {
      return;
    }

    Object.assign(providerFieldsByDisplayName, cacheEntries);
  }, [cacheEntries]);

  const allFields = React.useMemo(() => {
    // First try to resolve from the in-memory cache. We support both the
    // enum/display-name form and the raw provider slug (e.g. "petals").
    const cachedFields =
      providerFieldsByDisplayName[selectedProviderEnum] ?? providerFieldsByDisplayName[selectedProvider];
    if (cachedFields) {
      return cachedFields;
    }

    if (!providerMetadata) {
      return [];
    }

    const providerInfo = providerMetadata.find(
      (p) =>
        p.provider_display_name === selectedProviderEnum ||
        p.provider === selectedProvider ||
        p.litellm_provider === selectedProvider,
    );
    if (!providerInfo) {
      return [];
    }

    const mapped = providerInfo.credential_fields.map(mapFieldMetadataToUiField);
    providerFieldsByDisplayName[providerInfo.provider_display_name] = mapped;
    if (providerInfo.provider) {
      providerFieldsByDisplayName[providerInfo.provider] = mapped;
    }
    if (providerInfo.litellm_provider) {
      providerFieldsByDisplayName[providerInfo.litellm_provider] = mapped;
    }
    return mapped;
  }, [selectedProviderEnum, selectedProvider, providerMetadata]);

  const hasApiVersionField = React.useMemo(() => allFields.some((field) => field.key === "api_version"), [allFields]);
  const lastInferredApiVersionRef = React.useRef<string | null>(null);

  const handleApiBaseChange = React.useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      if (!hasApiVersionField) {
        return;
      }

      const apiVersion = getApiVersionFromApiBase(event.target.value);
      if (apiVersion) {
        lastInferredApiVersionRef.current = apiVersion;
        form.setValue("api_version", apiVersion);
        return;
      }

      if (form.getValues("api_version") === lastInferredApiVersionRef.current) {
        form.setValue("api_version", "");
      }
      lastInferredApiVersionRef.current = null;
    },
    [form, hasApiVersionField],
  );

  const renderFieldControl = (field: ProviderCredentialField, control: MountedFieldControlProps) => {
    if (field.type === "select") {
      return (
        <Select
          items={(field.options ?? []).map((option) => ({ value: option, label: option }))}
          value={(control.value as string | undefined) ?? field.defaultValue ?? null}
          onValueChange={control.onChange}
        >
          <SelectTrigger id={control.id} onBlur={control.onBlur} className="w-full">
            <SelectValue placeholder={field.placeholder} />
          </SelectTrigger>
          <SelectContent>
            {field.options?.map((option) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );
    }

    if (field.type === "upload") {
      return (
        <>
          <Button type="button" variant="outline" className="w-fit" onClick={() => credentialsFileRef.current?.click()}>
            <UploadIcon />
            Click to Upload
          </Button>
          <input
            ref={credentialsFileRef}
            id={control.id}
            type="file"
            accept=".json"
            className="sr-only"
            onBlur={control.onBlur}
            onChange={pickCredentialsFile(control.onChange)}
          />
        </>
      );
    }

    if (field.type === "textarea") {
      return (
        <Textarea
          id={control.id}
          value={control.value as string | undefined}
          onChange={control.onChange}
          onBlur={control.onBlur}
          placeholder={field.placeholder}
          defaultValue={field.defaultValue}
          rows={6}
          className="font-mono text-xs"
        />
      );
    }

    if (field.type === "password") {
      return (
        <PasswordInput
          id={control.id}
          value={control.value as string | undefined}
          onChange={control.onChange}
          onBlur={control.onBlur}
          placeholder={field.placeholder}
          defaultValue={field.defaultValue}
        />
      );
    }

    return (
      <Input
        id={control.id}
        value={(control.value as string | undefined) ?? undefined}
        onBlur={control.onBlur}
        placeholder={field.placeholder}
        type="text"
        defaultValue={field.defaultValue}
        onChange={(event) => {
          control.onChange(event);
          if (field.key === "api_base") {
            handleApiBaseChange(event);
          }
        }}
      />
    );
  };

  return (
    <>
      {isLoading && allFields.length === 0 && <p className="text-sm mb-2">Loading provider fields...</p>}
      {loadError && allFields.length === 0 && (
        <p className="text-sm mb-2 text-destructive">
          {loadError instanceof Error ? loadError.message : "Failed to load provider credential fields"}
        </p>
      )}
      {allFields.map((field) => (
        <React.Fragment key={field.key}>
          <MountedFormField
            label={field.tooltip ? labelWithHint(field.label, field.tooltip) : field.label}
            name={field.key}
            required={field.required}
            rules={field.required ? { validate: { required: requiredRule("Required") } } : undefined}
            className={field.key === "vertex_credentials" ? "mb-0" : "mb-4"}
          >
            {(control) => renderFieldControl(field, control)}
          </MountedFormField>

          {/* Special case for Vertex Credentials help text */}
          {field.key === "vertex_credentials" && (
            <p className="text-sm mb-3 mt-1">Give a gcp service account(.json file)</p>
          )}

          {/* Special case for Azure Base Model help text */}
          {field.key === "base_model" && (
            <div className="grid grid-cols-24">
              <p className="col-start-11 col-span-10 text-sm mb-2">
                The actual model your azure deployment uses. Used for accurate cost tracking. Select name from{" "}
                <a
                  href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline-offset-4 hover:underline"
                >
                  here
                </a>
              </p>
            </div>
          )}
        </React.Fragment>
      ))}
    </>
  );
};

export default ProviderSpecificFields;
