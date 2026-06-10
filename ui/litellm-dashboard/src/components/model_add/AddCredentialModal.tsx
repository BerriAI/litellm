import { TextInput } from "@tremor/react";
import { Select as AntdSelect, Button, Form, Modal, Tooltip, Typography } from "antd";
import type { UploadProps } from "antd/es/upload";
import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { Providers, providerLogoMap } from "../provider_info_helpers";
const { Link } = Typography;

interface AddCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  onAddCredential: (values: any) => void;
  uploadProps: UploadProps;
}

const AddCredentialsModal: React.FC<AddCredentialsModalProps> = ({ open, onCancel, onAddCredential, uploadProps }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.OpenAI);

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
      title={t("modelAdd.addCredentialModal.title")}
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
          label={t("modelAdd.addCredentialModal.credentialNameLabel")}
          name="credential_name"
          rules={[{ required: true, message: t("modelAdd.addCredentialModal.credentialNameRequired") }]}
        >
          <TextInput placeholder={t("modelAdd.addCredentialModal.credentialNamePlaceholder")} />
        </Form.Item>

        {/* Provider Selection */}
        <Form.Item
          rules={[{ required: true, message: t("common.required") }]}
          label={t("modelAdd.addCredentialModal.providerLabel")}
          name="custom_llm_provider"
          tooltip={t("modelAdd.addCredentialModal.providerTooltip")}
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
          <Tooltip title={t("modelAdd.addCredentialModal.needHelpTooltip")}>
            <Link href="https://github.com/BerriAI/litellm/issues">{t("modelAdd.addCredentialModal.needHelp")}</Link>
          </Tooltip>

          <div>
            <Button
              onClick={() => {
                onCancel();
                form.resetFields();
              }}
              style={{ marginRight: 10 }}
            >
              {t("common.cancel")}
            </Button>
            <Button htmlType="submit">{t("modelAdd.addCredentialModal.addCredential")}</Button>
          </div>
        </div>
      </Form>
    </Modal>
  );
};

export default AddCredentialsModal;
