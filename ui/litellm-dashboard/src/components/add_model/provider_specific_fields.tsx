import React from "react";
import { Form, Select } from "antd";
import { TextInput, Text } from "@tremor/react";
import { Row, Col, Typography, Button as Button2, Upload, UploadProps } from "antd";
import { UploadOutlined } from "@ant-design/icons";
import { provider_map, Providers } from "../provider_info_helpers";
import {
  CredentialItem,
  ProviderCreateInfo,
  ProviderCredentialFieldMetadata,
  getProviderCreateMetadata,
} from "../networking";
const { Link } = Typography;

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
  type?: "text" | "password" | "select" | "upload";
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

const ProviderSpecificFields: React.FC<ProviderSpecificFieldsProps> = ({ selectedProvider, uploadProps }) => {
  const selectedProviderEnum = Providers[selectedProvider as keyof typeof Providers] as Providers;
  const form = Form.useFormInstance(); // Get form instance from context

  const [providerMetadata, setProviderMetadata] = React.useState<ProviderCreateInfo[] | null>(null);
  const [isLoading, setIsLoading] = React.useState<boolean>(false);
  const [loadError, setLoadError] = React.useState<string | null>(null);

  React.useEffect(() => {
    const hasCachedFields = Object.keys(providerFieldsByDisplayName).length > 0;
    if (hasCachedFields) {
      // We already have fields cached globally; no need to refetch.
      // This is important so we can reuse credential field definitions
      // across mounts and in non-React helpers.
      return;
    }

    let isMounted = true;

    const fetchProviderFields = async () => {
      setIsLoading(true);
      setLoadError(null);
      try {
        const metadata = await getProviderCreateMetadata();
        if (!isMounted) {
          return;
        }
        setProviderMetadata(metadata);

        // Populate cache keyed by provider display name and identifiers
        metadata.forEach((providerInfo) => {
          const displayName = providerInfo.provider_display_name;
          const mappedFields = providerInfo.credential_fields.map(mapFieldMetadataToUiField);

          // Primary key: human-readable display name
          providerFieldsByDisplayName[displayName] = mappedFields;

          // Also cache by backend identifiers so lookups by provider slug work
          if (providerInfo.provider) {
            providerFieldsByDisplayName[providerInfo.provider] = mappedFields;
          }
          if (providerInfo.litellm_provider) {
            providerFieldsByDisplayName[providerInfo.litellm_provider] = mappedFields;
          }
        });
      } catch (error) {
        console.error("Failed to load provider credential fields:", error);
        if (isMounted) {
          setLoadError("Failed to load provider credential fields");
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchProviderFields();

    return () => {
      isMounted = false;
    };
  }, []);

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
            form.setFieldsValue({ vertex_credentials: jsonStr });
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
            <Text className="mb-2 text-red-500">{loadError}</Text>
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
                  // First call the original onChange
                  if (uploadProps?.onChange) {
                    uploadProps.onChange(info);
                  }

                  // Check the field value after a short delay
                  setTimeout(() => {
                    const value = form.getFieldValue(field.key);
                    console.log(`${field.key} value after upload:`, JSON.stringify(value));
                  }, 500);
                }}
              >
                <Button2 icon={<UploadOutlined />}>Click to Upload</Button2>
              </Upload>
            ) : (
              <TextInput
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
