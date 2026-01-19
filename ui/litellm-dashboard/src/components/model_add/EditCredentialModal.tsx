import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { TextInput } from "@tremor/react";
import { Select as AntdSelect, Button, Form, Modal, Tooltip, Typography } from "antd";
import type { UploadProps } from "antd/es/upload";
import { useEffect, useMemo, useState } from "react";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { ProviderLogo } from "../molecules/models/ProviderLogo";
import { CredentialItem, type ProviderCreateInfo } from "../networking";
import { Providers } from "../provider_info_helpers";
const { Link } = Typography;

interface EditCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  onUpdateCredential: (values: any) => void;
  uploadProps: UploadProps;
  existingCredential: CredentialItem | null;
}

export default function EditCredentialsModal({
  open,
  onCancel,
  onUpdateCredential,
  uploadProps,
  existingCredential,
}: EditCredentialsModalProps) {
  const [form] = Form.useForm();
  const [selectedProvider, setSelectedProvider] = useState<string | Providers>(Providers.Anthropic);

  const {
    data: providerMetadata,
    isLoading: isProviderMetadataLoading,
    error: providerMetadataError,
  } = useProviderFields();

  const sortedProviderMetadata: ProviderCreateInfo[] = useMemo(() => {
    if (!providerMetadata || !Array.isArray(providerMetadata)) {
      return [];
    }
    return [...providerMetadata].sort((a, b) => a.provider_display_name.localeCompare(b.provider_display_name));
  }, [providerMetadata]);

  const providerMetadataErrorText = providerMetadataError
    ? providerMetadataError instanceof Error
      ? providerMetadataError.message
      : "Failed to load providers"
    : null;

  const handleSubmit = (values: any) => {
    const filteredValues = Object.entries(values).reduce((acc, [key, value]) => {
      if (value !== "" && value !== undefined && value !== null) {
        acc[key] = value;
      }
      return acc;
    }, {} as any);
    onUpdateCredential(filteredValues);
    form.resetFields();
  };

  useEffect(() => {
    if (existingCredential) {
      // Spread all credential_values dynamically, converting undefined/null to null for form compatibility
      const credentialValues = Object.entries(existingCredential.credential_values || {}).reduce(
        (acc, [key, value]) => {
          acc[key] = value ?? null;
          return acc;
        },
        {} as Record<string, any>,
      );

      form.setFieldsValue({
        credential_name: existingCredential.credential_name,
        custom_llm_provider: existingCredential.credential_info.custom_llm_provider,
        ...credentialValues,
      });
      setSelectedProvider(existingCredential.credential_info.custom_llm_provider);
    }
  }, [existingCredential]);

  return (
    <Modal
      title="Edit Credential"
      open={open}
      onCancel={() => {
        onCancel();
        form.resetFields();
      }}
      footer={null}
      width={600}
      destroyOnHidden={true}
    >
      <Form form={form} onFinish={handleSubmit} layout="vertical">
        {/* Credential Name */}
        <Form.Item
          label="Credential Name:"
          name="credential_name"
          rules={[{ required: true, message: "Credential name is required" }]}
          initialValue={existingCredential?.credential_name}
        >
          <TextInput
            placeholder="Enter a friendly name for these credentials"
            disabled={existingCredential?.credential_name ? true : false}
          />
        </Form.Item>

        {/* Provider Selection */}
        <Form.Item
          rules={[{ required: true, message: "Required" }]}
          label="Provider:"
          name="custom_llm_provider"
          tooltip="Helper to auto-populate provider specific fields"
        >
          <AntdSelect
            showSearch
            loading={isProviderMetadataLoading}
            placeholder={isProviderMetadataLoading ? "Loading providers..." : "Select a provider"}
            optionFilterProp="data-label"
            onChange={(value) => {
              setSelectedProvider(value);
              form.setFieldValue("custom_llm_provider", value);
            }}
          >
            {providerMetadataErrorText && sortedProviderMetadata.length === 0 && (
              <AntdSelect.Option key="__error" value="">
                {providerMetadataErrorText}
              </AntdSelect.Option>
            )}
            {sortedProviderMetadata.map((providerInfo) => {
              const displayName = providerInfo.provider_display_name;
              const providerKey = providerInfo.provider;

              return (
                <AntdSelect.Option key={providerKey} value={providerKey} data-label={displayName}>
                  <div className="flex items-center space-x-2">
                    <ProviderLogo provider={providerKey} className="w-5 h-5" />
                    <span>{displayName}</span>
                  </div>
                </AntdSelect.Option>
              );
            })}
          </AntdSelect>
        </Form.Item>

        <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />

        {/* Modal Footer */}
        <div className="flex justify-between items-center">
          <Tooltip title="Get help on our github">
            <Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Link>
          </Tooltip>

          <div>
            <Button
              onClick={() => {
                onCancel();
                form.resetFields();
              }}
              style={{ marginRight: 10 }}
            >
              Cancel
            </Button>
            <Button htmlType="submit">{"Update Credential"}</Button>
          </div>
        </div>
      </Form>
    </Modal>
  );
}
