import { TextInput } from "@tremor/react";
import { Select as AntdSelect, Button, Form, Modal, Tooltip, Typography } from "antd";
import type { UploadProps } from "antd/es/upload";
import React, { useCallback, useMemo, useState } from "react";
import { credentialCreateCall } from "@/components/networking";
import ProviderSpecificFields from "../add_model/provider_specific_fields";
import { Providers, providerLogoMap } from "../provider_info_helpers";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useProviderFields } from "@/app/(dashboard)/hooks/providers/useProviderFields";
import { useDeviceCodeFlow } from "@/hooks/useDeviceCodeFlow";
const { Link } = Typography;

interface AddCredentialsModalProps {
  open: boolean;
  onCancel: () => void;
  onAddCredential: (values: any) => void;
  uploadProps: UploadProps;
}

const AddCredentialsModal: React.FC<AddCredentialsModalProps> = ({
  open,
  onCancel,
  onAddCredential,
  uploadProps,
}) => {
  const [form] = Form.useForm();
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.OpenAI);
  const { accessToken } = useAuthorized();
  const { data: providerMetadata } = useProviderFields();

  const deviceCodeProviderInfo = useMemo(() => {
    if (!providerMetadata) return null;
    const info = providerMetadata.find(
      (p) =>
        p.provider === selectedProvider ||
        p.provider_display_name === Providers[selectedProvider as keyof typeof Providers],
    );
    if (info?.auth_flow === "device_code") return info;
    return null;
  }, [selectedProvider, providerMetadata]);
  const isDeviceCodeProvider = deviceCodeProviderInfo != null;

  const handleDeviceCodeSuccess = useCallback(
    async (apiKey: string, litellmProvider: string) => {
      const credentialName = form.getFieldValue("credential_name");
      if (!credentialName) {
        form.validateFields(["credential_name"]);
        throw new Error("Credential name required");
      }
      if (!accessToken) throw new Error("No access token");
      await credentialCreateCall(accessToken, {
        credential_name: credentialName,
        credential_values: { api_key: apiKey },
        credential_info: { custom_llm_provider: litellmProvider },
      });
    },
    [form, accessToken],
  );

  const { state: deviceCodeState, start: startDeviceCode, reset: resetDeviceCode, renderUI: renderDeviceCodeFlow } = useDeviceCodeFlow({
    accessToken,
    providerInfo: deviceCodeProviderInfo,
    onSuccess: handleDeviceCodeSuccess,
  });

  const handleCancel = () => {
    resetDeviceCode();
    onCancel();
    form.resetFields();
  };

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

  const handleStartDeviceCode = async () => {
    const credentialName = form.getFieldValue("credential_name");
    if (!credentialName) {
      form.validateFields(["credential_name"]);
      return;
    }
    await startDeviceCode();
  };

  const handleSuccessClose = () => {
    resetDeviceCode();
    onCancel();
    form.resetFields();
  };

  const renderDeviceCodeSection = () => {
    if (deviceCodeState.phase === "idle") {
      return (
        <div className="text-center py-4">
          <Typography.Text className="block mb-4">
            {deviceCodeProviderInfo?.provider_display_name || "Provider"} uses OAuth Device Code authorization. Click below to start.
          </Typography.Text>
          <Button type="primary" onClick={handleStartDeviceCode}>
            Start {deviceCodeProviderInfo?.provider_display_name || "Provider"} Authorization
          </Button>
        </div>
      );
    }
    if (deviceCodeState.phase === "success") {
      return (
        <div className="text-center py-4">
          <Typography.Text className="block mb-4" type="success" style={{ fontSize: "1.1rem" }}>
            Credential created successfully!
          </Typography.Text>
          <Button type="primary" onClick={handleSuccessClose}>
            Done
          </Button>
        </div>
      );
    }
    return renderDeviceCodeFlow();
  };

  return (
    <Modal
      title="Add New Credential"
      open={open}
      onCancel={handleCancel}
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
              resetDeviceCode();
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

        {isDeviceCodeProvider ? (
          renderDeviceCodeSection()
        ) : (
          <>
            <ProviderSpecificFields selectedProvider={selectedProvider} uploadProps={uploadProps} />

            {/* Modal Footer */}
            <div className="flex justify-between items-center">
              <Tooltip title="Get help on our github">
                <Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Link>
              </Tooltip>

              <div>
                <Button
                  onClick={handleCancel}
                  style={{ marginRight: 10 }}
                >
                  Cancel
                </Button>
                <Button htmlType="submit">{"Add Credential"}</Button>
              </div>
            </div>
          </>
        )}
      </Form>
    </Modal>
  );
};

export default AddCredentialsModal;
