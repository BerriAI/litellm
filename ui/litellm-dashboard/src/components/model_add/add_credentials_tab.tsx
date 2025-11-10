import React, { useEffect, useState } from "react";
import { Form, Button, Tooltip, Typography, Select as AntdSelect, Modal } from "antd";
import type { UploadProps } from "antd/es/upload";
import { Providers, providerLogoMap } from "../provider_info_helpers";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { TextInput } from "@tremor/react";
import { CredentialItem } from "../networking";
const { Title, Link } = Typography;

interface AddCredentialsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onAddCredential: (values: any) => void;
  onUpdateCredential: (values: any) => void;
  uploadProps: UploadProps;
  addOrEdit: "add" | "edit";
  existingCredential: CredentialItem | null;
}

const AddCredentialsModal: React.FC<AddCredentialsModalProps> = ({
  isVisible,
  onCancel,
  onAddCredential,
  onUpdateCredential,
  uploadProps,
  addOrEdit,
  existingCredential,
}) => {
  const [form] = Form.useForm();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.OpenAI);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  const handleSubmit = (values: any) => {
    const filteredValues = Object.entries(values).reduce((acc, [key, value]) => {
      if (value !== "" && value !== undefined && value !== null) {
        acc[key] = value;
      }
      return acc;
    }, {} as any);
    if (addOrEdit === "add") {
      onAddCredential(filteredValues);
    } else {
      onUpdateCredential(filteredValues);
    }
    form.resetFields();
  };

  useEffect(() => {
    if (existingCredential) {
      form.setFieldsValue({
        credential_name: existingCredential.credential_name,
        custom_llm_provider: existingCredential.credential_info.custom_llm_provider,
        api_base: existingCredential.credential_values.api_base,
        api_version: existingCredential.credential_values.api_version,
        base_model: existingCredential.credential_values.base_model,
        api_key: existingCredential.credential_values.api_key,
      });
      setSelectedProvider(existingCredential.credential_info.custom_llm_provider as Providers);
    }
  }, [existingCredential]);

  return (
    <Modal
      title={addOrEdit === "add" ? "Add New Credential" : "Edit Credential"}
      visible={isVisible}
      onCancel={() => {
        onCancel();
        form.resetFields();
      }}
      footer={null}
      width={600}
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
            onChange={(value) => {
              setSelectedProvider(value as Providers);
              form.setFieldValue("custom_llm_provider", value);
            }}
          >
            {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => (
              <AntdSelect.Option key={providerEnum} value={providerEnum}>
                <div className="flex items-center space-x-2">
                  <img
                    src={providerLogoMap[providerDisplayName]}
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
            <Button htmlType="submit">{addOrEdit === "add" ? "Add Credential" : "Update Credential"}</Button>
          </div>
        </div>
      </Form>
    </Modal>
  );
};

export default AddCredentialsModal;
