import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Upload as UploadIcon } from "lucide-react";
import React from "react";
import { Controller, useFormContext } from "react-hook-form";
import type { UploadProps } from "./add_model_upload_types";
import {
  CredentialItem,
  ProviderCredentialFieldMetadata,
} from "../networking";
import { provider_map, Providers } from "../provider_info_helpers";

interface ProviderSpecificFieldsProps {
  selectedProvider: Providers;
  uploadProps?: UploadProps;
}

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

const mapFieldMetadataToUiField = (
  field: ProviderCredentialFieldMetadata,
): ProviderCredentialField => {
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
const providerFieldsByDisplayName: Record<string, ProviderCredentialField[]> =
  {};

export const createCredentialFromModel = (
  provider: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelData: any,
): CredentialItem => {
  const enumKey = Object.keys(provider_map).find(
    (key) => provider_map[key].toLowerCase() === provider.toLowerCase(),
  );
  if (!enumKey) {
    throw new Error(`Provider ${provider} not found in provider_map`);
  }
  const providerDisplayName = Providers[enumKey as keyof typeof Providers];
  const providerFields = providerFieldsByDisplayName[providerDisplayName] || [];
  const credentialValues: object = {};

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

const ProviderSpecificFields: React.FC<ProviderSpecificFieldsProps> = ({
  selectedProvider,
  uploadProps,
}) => {
  const selectedProviderEnum = Providers[
    selectedProvider as keyof typeof Providers
  ] as Providers;
  const { control, setValue, formState } = useFormContext();

  const {
    data: providerMetadata,
    isLoading,
    error: loadError,
  } = useProviderFields();

  // Memoize the expensive cache computation
  const cacheEntries = React.useMemo(() => {
    if (!providerMetadata) {
      return null;
    }

    const entries: Record<string, ProviderCredentialField[]> = {};
    providerMetadata.forEach((providerInfo) => {
      const displayName = providerInfo.provider_display_name;
      const mappedFields = providerInfo.credential_fields.map(
        mapFieldMetadataToUiField,
      );

      entries[displayName] = mappedFields;

      if (providerInfo.provider) {
        entries[providerInfo.provider] = mappedFields;
      }
      if (providerInfo.litellm_provider) {
        entries[providerInfo.litellm_provider] = mappedFields;
      }
    });
    return entries;
  }, [providerMetadata]);

  React.useEffect(() => {
    if (!cacheEntries) {
      return;
    }
    Object.assign(providerFieldsByDisplayName, cacheEntries);
  }, [cacheEntries]);

  const allFields = React.useMemo(() => {
    const cachedFields =
      providerFieldsByDisplayName[selectedProviderEnum] ??
      providerFieldsByDisplayName[selectedProvider];
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

  const fileInputRef = React.useRef<HTMLInputElement | null>(null);

  const handleFilePick = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.type === "application/json") {
      const text = await file.text();
      setValue("vertex_credentials", text, { shouldValidate: true });
    }
    // Delegate to consumer-provided upload props if present (kept for parity
    // with the antd Upload onChange callback).
    if (uploadProps?.onChange) {
      uploadProps.onChange({
        file: {
          name: file.name,
          status: "done",
          type: file.type,
        } as unknown as Parameters<NonNullable<UploadProps["onChange"]>>[0]["file"],
      });
    }
    // Reset the native input so the same file can be selected again.
    e.target.value = "";
  };

  return (
    <>
      {isLoading && allFields.length === 0 && (
        <div className="mb-2">Loading provider fields...</div>
      )}
      {loadError && allFields.length === 0 && (
        <div className="mb-2 text-destructive">
          {loadError instanceof Error
            ? loadError.message
            : "Failed to load provider credential fields"}
        </div>
      )}
      {allFields.map((field) => {
        const fieldError =
          (formState.errors as Record<string, { message?: string }>)[field.key];
        const requiredRule = field.required
          ? { required: "Required" as const }
          : {};

        return (
          <React.Fragment key={field.key}>
            <div
              className={`grid grid-cols-24 gap-2 mb-4 ${
                field.key === "vertex_credentials" ? "mb-0" : ""
              }`}
            >
              <Label
                className="col-span-10 pt-2"
                title={field.tooltip}
                htmlFor={`provider-field-${field.key}`}
              >
                {field.label}
                {field.required && (
                  <span className="text-destructive ml-1">*</span>
                )}
              </Label>
              <div className="col-span-14">
                {field.type === "select" ? (
                  <Controller
                    control={control}
                    name={field.key}
                    defaultValue={field.defaultValue ?? ""}
                    rules={requiredRule}
                    render={({ field: controllerField }) => (
                      <Select
                        value={(controllerField.value as string) || ""}
                        onValueChange={controllerField.onChange}
                      >
                        <SelectTrigger id={`provider-field-${field.key}`}>
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
                    )}
                  />
                ) : field.type === "upload" ? (
                  <>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".json"
                      hidden
                      onChange={handleFilePick}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <UploadIcon className="h-4 w-4" />
                      Click to Upload
                    </Button>
                  </>
                ) : field.type === "textarea" ? (
                  <Controller
                    control={control}
                    name={field.key}
                    defaultValue={field.defaultValue ?? ""}
                    rules={requiredRule}
                    render={({ field: controllerField }) => (
                      <Textarea
                        id={`provider-field-${field.key}`}
                        placeholder={field.placeholder}
                        rows={6}
                        className="font-mono text-xs"
                        {...controllerField}
                        value={(controllerField.value as string) ?? ""}
                      />
                    )}
                  />
                ) : (
                  <Controller
                    control={control}
                    name={field.key}
                    defaultValue={field.defaultValue ?? ""}
                    rules={requiredRule}
                    render={({ field: controllerField }) => (
                      <Input
                        id={`provider-field-${field.key}`}
                        placeholder={field.placeholder}
                        type={field.type === "password" ? "password" : "text"}
                        {...controllerField}
                        value={(controllerField.value as string) ?? ""}
                      />
                    )}
                  />
                )}
                {fieldError?.message && (
                  <p className="text-sm text-destructive mt-1">
                    {String(fieldError.message)}
                  </p>
                )}
              </div>
            </div>

            {/* Special case for Vertex Credentials help text */}
            {field.key === "vertex_credentials" && (
              <div className="grid grid-cols-24 gap-2">
                <div className="col-span-24">
                  <p className="mb-3 mt-1">
                    Give a gcp service account(.json file)
                  </p>
                </div>
              </div>
            )}

            {/* Special case for Azure Base Model help text */}
            {field.key === "base_model" && (
              <div className="grid grid-cols-24 gap-2">
                <div className="col-span-10" />
                <div className="col-span-14">
                  <p className="mb-2">
                    The actual model your azure deployment uses. Used for
                    accurate cost tracking. Select name from{" "}
                    <a
                      href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-primary hover:text-primary/80 underline"
                    >
                      here
                    </a>
                  </p>
                </div>
              </div>
            )}
          </React.Fragment>
        );
      })}
    </>
  );
};

export default ProviderSpecificFields;
