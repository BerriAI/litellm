import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { UploadOutlined } from "@ant-design/icons";
import { Text, TextInput } from "@tremor/react";
import { Button as Button2, Col, Form, Input, Row, Select, Typography, Upload, UploadProps } from "antd";
import React from "react";
import { CredentialItem, ProviderCredentialFieldMetadata } from "../networking";
import { provider_map, Providers } from "../provider_info_helpers";
const { Link } = Typography;

interface ProviderSpecificFieldsProps {
  selectedProvider: Providers;
  uploadProps?: UploadProps;
  /**
   * Form rendering mode:
   * - "create" (default): standard create flow — `required` fields validate and
   *   the field metadata's placeholder/defaultValue are honored.
   * - "rotate": edit/rotate flow — all fields are optional, the placeholder is
   *   overridden to instruct the user to leave blank to keep the current value,
   *   and field-level defaults are suppressed so empty submissions stay empty
   *   (and are interpreted by the caller as "keep current").
   */
  mode?: "create" | "rotate";
  /**
   * Optional prefix applied to every `Form.Item.name` (and to the upload
   * setFieldsValue / getFieldValue lookups) registered by this component.
   *
   * Use this in edit flows where the parent form already registers
   * `Form.Item`s whose `name` collides with provider credential field keys
   * (for example, `ModelInfoView` has a top-level `organization` field that
   * collides with the OpenAI provider's `organization` credential field).
   * Without a prefix the two would share antd Form state, defeating the
   * "Leave blank to keep current value" contract.
   *
   * The parent reads each rotate-form value back as
   * `values[fieldNamePrefix + field.key]`.
   */
  fieldNamePrefix?: string;
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

export const createCredentialFromModel = (provider: string, modelData: any): CredentialItem => {
  console.log("provider", provider);
  console.log("modelData", modelData);
  const enumKey = Object.keys(provider_map).find((key) => provider_map[key].toLowerCase() === provider.toLowerCase());
  if (!enumKey) {
    throw new Error(`Provider ${provider} not found in provider_map`);
  }
  const providerDisplayName = Providers[enumKey as keyof typeof Providers];
  const providerFields = providerFieldsByDisplayName[providerDisplayName] || [];
  const credentialValues: object = {};

  console.log("providerFields", providerFields);

  // Go through each field defined for this provider
  providerFields.forEach((field) => {
    const value = modelData.litellm_params[field.key];
    console.log("field", field);
    console.log("value", value);
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

const ROTATE_PLACEHOLDER = "Leave blank to keep current value";

const ProviderSpecificFields: React.FC<ProviderSpecificFieldsProps> = ({
  selectedProvider,
  uploadProps,
  mode = "create",
  fieldNamePrefix = "",
}) => {
  const isRotateMode = mode === "rotate";
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
    beforeUpload: (file: any) => {
      if (file.type === "application/json") {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (e.target) {
            const jsonStr = e.target.result as string;
            console.log(`Setting field value from JSON, length: ${jsonStr.length}`);
            form.setFieldsValue({ [fieldNamePrefix + "vertex_credentials"]: jsonStr });
            console.log("Form values after setting:", form.getFieldsValue());
          }
        };
        reader.readAsText(file);
      }
      // Prevent upload
      return false;
    },
    onChange(info: any) {
      console.log("Upload onChange triggered in ProviderSpecificFields");
      console.log("Current form values:", form.getFieldsValue());

      if (info.file.status !== "uploading") {
        console.log(info.file, info.fileList);
      }
    },
  };

  return (
    <>
      {isLoading && allFields.length === 0 && (
        <Row>
          <Col span={24}>
            <Text className="mb-2">Loading provider fields...</Text>
          </Col>
        </Row>
      )}
      {loadError && allFields.length === 0 && (
        <Row>
          <Col span={24}>
            <Text className="mb-2 text-red-500">
              {loadError instanceof Error ? loadError.message : "Failed to load provider credential fields"}
            </Text>
          </Col>
        </Row>
      )}
      {allFields.map((field) => (
        <React.Fragment key={field.key}>
          <Form.Item
            label={field.label}
            name={fieldNamePrefix + field.key}
            rules={isRotateMode ? undefined : field.required ? [{ required: true, message: "Required" }] : undefined}
            tooltip={field.tooltip}
            className={field.key === "vertex_credentials" ? "mb-0" : undefined}
          >
            {field.type === "select" ? (
              <Select
                placeholder={isRotateMode ? ROTATE_PLACEHOLDER : field.placeholder}
                defaultValue={isRotateMode ? undefined : field.defaultValue}
              >
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
                  // First call the original onChange
                  if (uploadProps?.onChange) {
                    uploadProps.onChange(info);
                  }

                  // Check the field value after a short delay
                  setTimeout(() => {
                    const namespacedKey = fieldNamePrefix + field.key;
                    const value = form.getFieldValue(namespacedKey);
                    console.log(`${namespacedKey} value after upload:`, JSON.stringify(value));
                  }, 500);
                }}
              >
                <Button2 icon={<UploadOutlined />}>Click to Upload</Button2>
              </Upload>
            ) : field.type === "textarea" ? (
              <Input.TextArea
                placeholder={isRotateMode ? ROTATE_PLACEHOLDER : field.placeholder}
                defaultValue={isRotateMode ? undefined : field.defaultValue}
                rows={6}
                style={{ fontFamily: "monospace", fontSize: "12px" }}
              />
            ) : (
              <TextInput
                placeholder={isRotateMode ? ROTATE_PLACEHOLDER : field.placeholder}
                type={field.type === "password" ? "password" : "text"}
                defaultValue={isRotateMode ? undefined : field.defaultValue}
              />
            )}
          </Form.Item>

          {/* Special case for Vertex Credentials help text */}
          {field.key === "vertex_credentials" && (
            <Row>
              <Col>
                <Text className="mb-3 mt-1">Give a gcp service account(.json file)</Text>
              </Col>
            </Row>
          )}

          {/* Special case for Azure Base Model help text */}
          {field.key === "base_model" && (
            <Row>
              <Col span={10}></Col>
              <Col span={10}>
                <Text className="mb-2">
                  The actual model your azure deployment uses. Used for accurate cost tracking. Select name from{" "}
                  <Link
                    href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                    target="_blank"
                  >
                    here
                  </Link>
                </Text>
              </Col>
            </Row>
          )}
        </React.Fragment>
      ))}
    </>
  );
};

export default ProviderSpecificFields;
