import { TextInput } from "@tremor/react";
import { Select as AntdSelect, Button, Form, Modal, Tooltip, Typography } from "antd";
import type { UploadProps } from "antd/es/upload";
import { useEffect, useState } from "react";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { CredentialItem } from "../networking";
import { Providers, providerLogoMap } from "../provider_info_helpers";
import { resolveLogoSrc } from "@/lib/assetPaths";
import { resetCredentialFormOnProviderChange } from "./credential_form_helpers";

const { Link } = Typography;

interface CredentialModalProps {
  open: boolean;
  onCancel: () => void;
  onSubmit: (values: any) => void;
  uploadProps: UploadProps;
  mode: "add" | "edit";
  existingCredential?: CredentialItem | null;
}

export default function CredentialModal({
  open,
  onCancel,
  onSubmit,
  uploadProps,
  mode,
  existingCredential = null,
}: CredentialModalProps) {
  const isEdit = mode === "edit";
  const [form] = Form.useForm();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(isEdit ? Providers.Anthropic : Providers.OpenAI);

  const handleSubmit = (values: any) => {
    const filteredValues = Object.entries(values).reduce((acc, [key, value]) => {
      if (value !== "" && value !== undefined && value !== null) {
        acc[key] = value;
      }
      return acc;
    }, {} as any);
    onSubmit(filteredValues);
    form.resetFields();
  };

  useEffect(() => {
    if (!existingCredential) return;
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
    setSelectedProvider(existingCredential.credential_info.custom_llm_provider as Providers);
  }, [existingCredential]);

  const closeAndReset = () => {
    onCancel();
    form.resetFields();
  };

  return (
    <Modal
      title={isEdit ? "Edit Credential" : "Add New Credential"}
      open={open}
      onCancel={closeAndReset}
      footer={null}
      width={600}
      destroyOnHidden={isEdit}
    >
      <Form form={form} onFinish={handleSubmit} layout="vertical">
        <Form.Item
          label="Credential Name:"
          name="credential_name"
          rules={[{ required: true, message: "Credential name is required" }]}
          initialValue={existingCredential?.credential_name}
        >
          <TextInput placeholder="Enter a friendly name for these credentials" disabled={isEdit} />
        </Form.Item>

        <Form.Item
          rules={[{ required: true, message: "Required" }]}
          label="Provider:"
          name="custom_llm_provider"
          tooltip="Helper to auto-populate provider specific fields"
        >
          <AntdSelect
            showSearch
            onChange={(value) => {
              resetCredentialFormOnProviderChange(form, value as Providers, setSelectedProvider);
            }}
          >
            {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => (
              <AntdSelect.Option key={providerEnum} value={providerEnum}>
                <div className="flex items-center space-x-2">
                  <img
                    src={resolveLogoSrc(providerLogoMap[providerDisplayName])}
                    alt={`${providerEnum} logo`}
                    className="w-5 h-5"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement;
                      const parent = target.parentElement;
                      if (parent) {
                        const fallbackDiv = document.createElement("div");
                        fallbackDiv.className =
                          "w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs";
                        fallbackDiv.textContent = providerDisplayName.charAt(0);
                        parent.replaceChild(fallbackDiv, target);
                      }
                    }}
                  />
                  <span>{providerDisplayName}</span>
                </div>
              </AntdSelect.Option>
            ))}
          </AntdSelect>
        </Form.Item>

        <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />

        <div className="flex justify-between items-center">
          <Tooltip title="Get help on our github">
            <Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Link>
          </Tooltip>

          <div>
            <Button onClick={closeAndReset} style={{ marginRight: 10 }}>
              Cancel
            </Button>
            <Button htmlType="submit">{isEdit ? "Update Credential" : "Add Credential"}</Button>
          </div>
        </div>
      </Form>
    </Modal>
  );
}
