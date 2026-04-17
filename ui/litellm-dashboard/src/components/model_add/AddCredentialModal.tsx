import { TextInput } from "@tremor/react";
import { Select as AntdSelect, Button, Form, Modal, Tooltip, Typography } from "antd";
import type { UploadProps } from "antd/es/upload";
import React, { useState } from "react";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { Providers, providerLogoMap } from "../provider_info_helpers";
import ChatGPTLoginButton from "./ChatGPTLoginButton";
import CopilotLoginButton from "./CopilotLoginButton";
const { Link } = Typography;

interface AddCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  onAddCredential: (values: any) => void;
  uploadProps: UploadProps;
}

const AddCredentialsModal: React.FC<AddCredentialsModalProps> = ({ open, onCancel, onAddCredential, uploadProps }) => {
  const [form] = Form.useForm();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.OpenAI);
  const credentialName = Form.useWatch("credential_name", form);

  const handleSubmit = (values: any) => {
    const filteredValues = Object.entries(values).reduce((acc, [key, value]) => {
      if (value !== "" && value !== undefined && value !== null) {
        acc[key] = value;
      }
      return acc;
    }, {} as any);
    onAddCredential(filteredValues);
    form.resetFields();
  };

  return (
    <Modal
      title="Add New Credential"
      open={open}
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
        >
          <TextInput placeholder="Enter a friendly name for these credentials" />
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

        {/*
          `selectedProvider` holds the enum *key* string at runtime
          (``AntdSelect.Option value={providerEnum}`` binds the key, and the
          ``as Providers`` cast on setState is not enforced). Compare against
          the key names, not ``Providers.ChatGPT`` / ``Providers.GITHUB_COPILOT``
          which resolve to enum *values* and would never match.
        */}
        {(selectedProvider as unknown as keyof typeof Providers) === "ChatGPT" ? (
          <ChatGPTLoginButton
            credentialName={credentialName}
            onSuccess={() => {
              onCancel();
              form.resetFields();
            }}
          />
        ) : (selectedProvider as unknown as keyof typeof Providers) === "GITHUB_COPILOT" ? (
          <CopilotLoginButton
            credentialName={credentialName}
            onSuccess={() => {
              onCancel();
              form.resetFields();
            }}
          />
        ) : (
          <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />
        )}

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
            {(selectedProvider as unknown as keyof typeof Providers) !== "ChatGPT" &&
              (selectedProvider as unknown as keyof typeof Providers) !== "GITHUB_COPILOT" && (
                <Button htmlType="submit">{"Add Credential"}</Button>
              )}
          </div>
        </div>
      </Form>
    </Modal>
  );
};

export default AddCredentialsModal;
