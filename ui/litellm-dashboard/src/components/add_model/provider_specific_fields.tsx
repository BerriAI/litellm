import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Upload as UploadIcon } from "lucide-react";
import {
  Col,
  Form,
  Row,
  Select,
  Upload,
  UploadProps,
} from "antd";
import React from "react";
import { CredentialItem, ProviderCredentialFieldMetadata } from "../networking";
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

const ProviderSpecificFields: React.FC<ProviderSpecificFieldsProps> = ({ selectedProvider, uploadProps }) => {
  const selectedProviderEnum = Providers[selectedProvider as keyof typeof Providers] as Providers;
  const form = Form.useFormInstance(); // Get form instance from context

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

  const handleUpload = {
    name: "file",
    accept: ".json",
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    beforeUpload: (file: any) => {
      if (file.type === "application/json") {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (e.target) {
            const jsonStr = e.target.result as string;
            form.setFieldsValue({ vertex_credentials: jsonStr });
          }
        };
        reader.readAsText(file);
      }
      return false;
    },
  };

  return (
    <>
      {isLoading && allFields.length === 0 && (
        <Row>
          <Col span={24}>
            <p className="mb-2">Loading provider fields...</p>
          </Col>
        </Row>
      )}
      {loadError && allFields.length === 0 && (
        <Row>
          <Col span={24}>
            <p className="mb-2 text-destructive">
              {loadError instanceof Error
                ? loadError.message
                : "Failed to load provider credential fields"}
            </p>
          </Col>
        </Row>
      )}
      {allFields.map((field) => (
        <React.Fragment key={field.key}>
          <Form.Item
            label={field.label}
            name={field.key}
            rules={field.required ? [{ required: true, message: "Required" }] : undefined}
            tooltip={field.tooltip}
            className={field.key === "vertex_credentials" ? "mb-0" : undefined}
          >
            {field.type === "select" ? (
              <Select placeholder={field.placeholder} defaultValue={field.defaultValue}>
                {field.options?.map((option) => (
                  <Select.Option key={option} value={option}>
                    {option}
                  </Select.Option>
                ))}
              </Select>
            ) : field.type === "upload" ? (
              <Upload
                {...handleUpload}
                onChange={(info) => {
                  if (uploadProps?.onChange) {
                    uploadProps.onChange(info);
                  }
                }}
              >
                <Button type="button" variant="outline">
                  <UploadIcon className="h-4 w-4" />
                  Click to Upload
                </Button>
              </Upload>
            ) : field.type === "textarea" ? (
              <Textarea
                placeholder={field.placeholder}
                defaultValue={field.defaultValue}
                rows={6}
                className="font-mono text-xs"
              />
            ) : (
              <Input
                placeholder={field.placeholder}
                type={field.type === "password" ? "password" : "text"}
                defaultValue={field.defaultValue}
              />
            )}
          </Form.Item>

          {/* Special case for Vertex Credentials help text */}
          {field.key === "vertex_credentials" && (
            <Row>
              <Col>
                <p className="mb-3 mt-1">
                  Give a gcp service account(.json file)
                </p>
              </Col>
            </Row>
          )}

          {/* Special case for Azure Base Model help text */}
          {field.key === "base_model" && (
            <Row>
              <Col span={10}></Col>
              <Col span={10}>
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
              </Col>
            </Row>
          )}
        </React.Fragment>
      ))}
    </>
  );
};

export default ProviderSpecificFields;
